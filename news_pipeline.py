# news_pipeline.py - Apple-focused version with NewsAPI
"""
Milestone 4: News Feed Pipeline - Apple-focused version
- Fetch Apple-specific news from multiple sources
- Clean, deduplicate, and process text
- Extract entities, sentiment, and create embeddings
- Store in vector database
"""

import os
import json
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import re
from collections import defaultdict

# LangChain and OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()

# NLP libraries
from textblob import TextBlob  # For sentiment analysis
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

@dataclass
class NewsArticle:
    """Structured representation of a news article"""
    id: str
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    author: Optional[str] = None
    
    # Processed fields
    cleaned_content: Optional[str] = None
    entities: Optional[List[Dict]] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None
    
    # Metadata
    created_at: datetime = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class NewsAPIClient:
    """Client for fetching news from NewsAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def fetch_everything(
        self, 
        query: str, 
        domains: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100
    ) -> List[Dict]:
        """Fetch articles using the everything endpoint"""
        
        url = f"{self.base_url}/everything"
        
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size
        }
        
        if domains:
            params["domains"] = domains
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "ok":
                return data["articles"]
            else:
                print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

class RSSFeedClient:
    """Client for fetching Apple-focused news from RSS feeds"""
    
    def __init__(self):
        # Apple-focused RSS feeds
        self.feeds = {
            "apple_newsroom": "https://www.apple.com/newsroom/rss-feed.rss",
            "macrumors": "https://feeds.macrumors.com/MacRumors-All",
            "9to5mac": "https://9to5mac.com/feed/",
            "appleinsider": "https://appleinsider.com/rss/news/",
            "techcrunch_apple": "https://techcrunch.com/tag/apple/feed/",
            "ars_technica": "http://feeds.arstechnica.com/arstechnica/index",
            "reuters_tech": "https://feeds.reuters.com/reuters/technologyNews",
            "bloomberg_tech": "https://feeds.bloomberg.com/technology/news.rss",
        }
    
    def fetch_feed(self, feed_url: str) -> List[Dict]:
        """Fetch articles from a single RSS feed"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries:
                # Safely extract fields with proper defaults
                title = entry.get("title", "")
                content = entry.get("summary", entry.get("description", ""))
                url = entry.get("link", "")
                published_date = entry.get("published", "")
                author = entry.get("author", None)
                
                # Get feed title safely
                source = "Unknown RSS Feed"
                if hasattr(feed, 'feed') and hasattr(feed.feed, 'title'):
                    source = feed.feed.title or "Unknown RSS Feed"
                
                # Only include articles with at least title and URL
                if title and url:
                    article = {
                        "title": title,
                        "content": content or "",  # Ensure content is never None
                        "url": url,
                        "published_date": published_date,
                        "author": author,
                        "source": source
                    }
                    articles.append(article)
                
            return articles
            
        except Exception as e:
            print(f"Error fetching RSS feed {feed_url}: {e}")
            return []
    
    def fetch_all_feeds(self) -> List[Dict]:
        """Fetch articles from all configured RSS feeds"""
        all_articles = []
        
        for feed_name, feed_url in self.feeds.items():
            print(f"Fetching {feed_name}...")
            articles = self.fetch_feed(feed_url)
            
            # Add feed name to source
            for article in articles:
                article["source"] = f"{article['source']} ({feed_name})"
                
            all_articles.extend(articles)
            print(f"  Found {len(articles)} articles")
            
        return all_articles

class NewsProcessor:
    """Process raw news articles into structured format"""
    
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        # Try to load spaCy model for entity extraction
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        return text.strip()
    
    def extract_entities_spacy(self, text: str) -> List[Dict]:
        """Extract entities using spaCy"""
        if not self.nlp or not text:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0  # spaCy doesn't provide confidence scores by default
                })
        
        return entities
    
    def extract_entities_llm(self, text: str) -> List[Dict]:
        """Extract entities using LLM for better accuracy"""
        if not text or len(text.strip()) < 10:
            return []
            
        prompt_template = """
        Extract key entities from this Apple-related news text.
        
        Focus SPECIFICALLY on:
        - Apple Inc. and its business units
        - Apple products (iPhone, iPad, Mac, Apple Watch, etc.)
        - Apple services (App Store, iCloud, Apple Pay, etc.)
        - Apple executives and employees
        - Apple suppliers and manufacturing partners (Foxconn, TSMC, etc.)
        - Apple competitors (Samsung, Google, Microsoft, etc.)
        - Geographic locations relevant to Apple operations
        - Technologies used by Apple
        - Regulatory bodies affecting Apple
        
        TEXT:
        {text}
        
        Return a JSON list of entities with format:
        [
            {{"text": "entity name", "type": "COMPANY|PRODUCT|PERSON|LOCATION|TECHNOLOGY", "relevance": "HIGH|MEDIUM|LOW"}}
        ]
        
        Return ONLY valid JSON:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({"text": text[:3000]}).content  # Limit text length
            
            # Clean up the result - sometimes LLM adds extra text
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            entities = json.loads(result)
            
            # Standardize format
            standardized = []
            for ent in entities:
                if isinstance(ent, dict) and "text" in ent:
                    standardized.append({
                        "text": ent["text"],
                        "label": ent.get("type", "UNKNOWN"),
                        "relevance": ent.get("relevance", "MEDIUM"),
                        "confidence": 0.8  # LLM extraction confidence
                    })
            
            return standardized
            
        except Exception as e:
            print(f"Error in LLM entity extraction: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment using TextBlob"""
        if not text:
            return 0.0, "neutral"
            
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            
            # Convert to label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
                
            return polarity, label
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.0, "neutral"
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text using OpenAI"""
        try:
            if not text or len(text.strip()) < 5:
                return None
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None
    
    def safe_string(self, value: Any) -> str:
        """Safely convert any value to string, handling None"""
        if value is None:
            return ""
        return str(value)
    
    def process_article(self, raw_article: Dict) -> NewsArticle:
        """Process a single raw article into a NewsArticle object"""
        
        try:
            # Safely extract fields
            title = self.safe_string(raw_article.get("title", ""))
            content = self.safe_string(raw_article.get("content", ""))
            url = self.safe_string(raw_article.get("url", ""))
            source = self.safe_string(raw_article.get("source", ""))
            author = raw_article.get("author")  # Can be None
            
            # Skip articles with no meaningful content
            if not title and not content:
                raise ValueError("Article has no title or content")
            
            # Generate unique ID
            content_for_hash = f"{title}{url}"
            content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
            
            # Parse date
            published_date = raw_article.get("published_date")
            if isinstance(published_date, str) and published_date:
                try:
                    # Try different date formats
                    for fmt in ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                        try:
                            published_date = datetime.strptime(published_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        published_date = datetime.now()
                except:
                    published_date = datetime.now()
            elif not published_date:
                published_date = datetime.now()
            
            # Create initial article object
            article = NewsArticle(
                id=content_hash,
                title=title,
                content=content,
                url=url,
                source=source,
                published_date=published_date,
                author=author
            )
            
            # Clean content - safely combine title and content
            title_text = title if title else ""
            content_text = content if content else ""
            
            if title_text and content_text:
                full_text = f"{title_text}. {content_text}"
            elif title_text:
                full_text = title_text
            elif content_text:
                full_text = content_text
            else:
                full_text = ""
            
            article.cleaned_content = self.clean_text(full_text)
            
            # Skip if no meaningful text after cleaning
            if len(article.cleaned_content) < 10:
                raise ValueError("Article has no meaningful content after cleaning")
            
            # Extract entities (try LLM first, fallback to spaCy)
            entities = self.extract_entities_llm(article.cleaned_content)
            if not entities and self.nlp:
                entities = self.extract_entities_spacy(article.cleaned_content)
            article.entities = entities
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self.analyze_sentiment(article.cleaned_content)
            article.sentiment_score = sentiment_score
            article.sentiment_label = sentiment_label
            
            # Create embedding
            article.embedding = self.create_embedding(article.cleaned_content)
            
            # Mark as processed
            article.processed_at = datetime.now()
            
            return article
            
        except Exception as e:
            print(f"Error in process_article: {e}")
            # Return minimal article for debugging
            return NewsArticle(
                id=f"error_{datetime.now().timestamp()}",
                title=self.safe_string(raw_article.get("title", "Error Article")),
                content=f"Error processing: {str(e)}",
                url=self.safe_string(raw_article.get("url", "")),
                source=self.safe_string(raw_article.get("source", "")),
                published_date=datetime.now()
            )

class NewsDatabase:
    """Vector database for storing and querying news articles"""
    
    def __init__(self, db_path: str = "./news_db"):
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="news_articles",
                metadata={"description": "Apple-related news articles with embeddings"}
            )
        except Exception as e:
            print(f"Warning: Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_article(self, article: NewsArticle) -> bool:
        """Add a single article to the database"""
        if not self.collection:
            print("Database not available")
            return False
            
        try:
            if not article.embedding:
                print(f"Warning: Article {article.id} has no embedding, skipping")
                return False
            
            # Prepare metadata
            metadata = {
                "title": article.title[:500],  # Limit length
                "url": article.url[:500],
                "source": article.source[:200],
                "published_date": article.published_date.isoformat(),
                "author": (article.author or "")[:200],
                "sentiment_score": article.sentiment_score or 0.0,
                "sentiment_label": article.sentiment_label or "neutral",
                "relevance_score": article.relevance_score or 0.0,
                "processed_at": article.processed_at.isoformat() if article.processed_at else "",
                "entities": json.dumps(article.entities or [])[:1000]  # Limit length
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[article.embedding],
                documents=[article.cleaned_content[:2000]],  # Limit document length
                metadatas=[metadata],
                ids=[article.id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding article to database: {e}")
            return False
    
    def add_articles(self, articles: List[NewsArticle]) -> int:
        """Add multiple articles to the database"""
        added_count = 0
        
        for article in articles:
            if self.add_article(article):
                added_count += 1
        
        return added_count
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """Search for similar articles using vector similarity"""
        if not self.collection:
            return {"documents": [], "metadatas": [], "distances": []}
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            return results
        except Exception as e:
            print(f"Error searching database: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def get_recent_articles(self, days: int = 7) -> Dict:
        """Get articles from the last N days"""
        if not self.collection:
            return {"documents": [], "metadatas": [], "distances": []}
            
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Get all articles (ChromaDB doesn't support date range queries well)
            results = self.collection.get()
            
            # Filter by date manually
            recent_docs = []
            recent_metadata = []
            recent_ids = []
            
            if results.get("metadatas"):
                for i, metadata in enumerate(results["metadatas"]):
                    pub_date = metadata.get("published_date", "")
                    if pub_date >= cutoff_date:
                        recent_docs.append(results["documents"][i])
                        recent_metadata.append(metadata)
                        recent_ids.append(results["ids"][i])
            
            return {
                "documents": recent_docs,
                "metadatas": recent_metadata,
                "ids": recent_ids,
                "distances": []
            }
        except Exception as e:
            print(f"Error getting recent articles: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def get_article_count(self) -> int:
        """Get total number of articles in database"""
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except:
            return 0

class NewsPipeline:
    """Main pipeline orchestrating Apple news collection and processing"""
    
    def __init__(
        self, 
        openai_api_key: str,
        newsapi_key: Optional[str] = None,
        db_path: str = "./news_db"
    ):
        self.processor = NewsProcessor(openai_api_key)
        self.database = NewsDatabase(db_path)
        
        # Initialize news sources
        self.rss_client = RSSFeedClient()
        self.newsapi_client = NewsAPIClient(newsapi_key) if newsapi_key else None
        
        # Apple-focused search terms for NewsAPI
        self.apple_queries = [
            "Apple Inc",
            "iPhone OR iPad OR Mac OR MacBook",
            "Tim Cook OR Apple CEO",
            "Foxconn Apple OR TSMC Apple",
            "Apple supply chain",
            "Apple earnings OR Apple revenue",
            "iOS OR macOS OR watchOS",
            "App Store OR iCloud",
            "Apple Watch OR AirPods",
            "Apple Silicon OR M1 OR M2 OR M3"
        ]
    
    def fetch_news(self, max_articles: int = 200) -> List[Dict]:
        """Fetch Apple-focused news from all configured sources"""
        all_articles = []
        
        # Fetch from Apple-focused RSS feeds
        print("Fetching from Apple-focused RSS feeds...")
        rss_articles = self.rss_client.fetch_all_feeds()
        all_articles.extend(rss_articles)
        
        # Fetch from NewsAPI if available
        if self.newsapi_client:
            print("Fetching from NewsAPI with Apple-specific queries...")
            from_date = datetime.now() - timedelta(days=3)  # Last 3 days for more focused results
            
            for query in self.apple_queries[:5]:  # Use more queries since we're Apple-focused
                print(f"  Searching for: {query}")
                articles = self.newsapi_client.fetch_everything(
                    query=query,
                    from_date=from_date,
                    page_size=20  # Limit per query
                )
                
                # Convert NewsAPI format to our format
                for article in articles:
                    formatted_article = {
                        "title": article.get("title", ""),
                        "content": article.get("description", "") or article.get("content", ""),
                        "url": article.get("url", ""),
                        "published_date": article.get("publishedAt", ""),
                        "author": article.get("author"),
                        "source": article.get("source", {}).get("name", "NewsAPI")
                    }
                    all_articles.append(formatted_article)
                
                print(f"    Found {len(articles)} articles")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            url = article.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print(f"Found {len(unique_articles)} unique articles after deduplication")
        return unique_articles[:max_articles]
    
    def process_and_store(self, raw_articles: List[Dict]) -> List[NewsArticle]:
        """Process raw articles and store them in the database"""
        processed_articles = []
        
        print(f"Processing {len(raw_articles)} articles...")
        
        for i, raw_article in enumerate(raw_articles):
            try:
                print(f"Processing article {i+1}/{len(raw_articles)}: {raw_article.get('title', 'No title')[:50]}...")
                
                article = self.processor.process_article(raw_article)
                
                # Only add if processing was successful (has meaningful content)
                if article.cleaned_content and len(article.cleaned_content) > 10:
                    processed_articles.append(article)
                    
                    # Add to database
                    if self.database.add_article(article):
                        print(f"  ✅ Processed and stored successfully")
                    else:
                        print(f"  ⚠️ Processed but failed to store in database")
                else:
                    print(f"  ❌ Skipped - no meaningful content")
                
            except Exception as e:
                print(f"Error processing article {i+1}: {e}")
                continue
        
        print(f"Successfully processed and stored {len(processed_articles)} articles")
        return processed_articles
    
    def run_pipeline(self, max_articles: int = 100) -> Dict[str, Any]:
        """Run the complete Apple news pipeline"""
        start_time = datetime.now()
        
        print("=== Starting Apple News Pipeline ===")
        
        # Fetch news
        raw_articles = self.fetch_news(max_articles)
        
        # Process and store
        processed_articles = self.process_and_store(raw_articles)
        
        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "articles_fetched": len(raw_articles),
            "articles_processed": len(processed_articles),
            "total_articles_in_db": self.database.get_article_count(),
            "success_rate": len(processed_articles) / len(raw_articles) if raw_articles else 0
        }
        
        print(f"=== Apple News Pipeline Complete ===")
        print(f"Processed {summary['articles_processed']}/{summary['articles_fetched']} articles in {duration:.1f} seconds")
        print(f"Total articles in database: {summary['total_articles_in_db']}")
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configuration (you'll need to set these environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Set this to your NewsAPI key
    
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
        
    if not NEWSAPI_KEY:
        print("Warning: NEWSAPI_KEY not set - will only use RSS feeds")
    
    # Initialize pipeline
    pipeline = NewsPipeline(
        openai_api_key=OPENAI_API_KEY,
        newsapi_key=NEWSAPI_KEY,
        db_path="./apple_news_db"
    )
    
    # Run pipeline
    result = pipeline.run_pipeline(max_articles=50)
    print(json.dumps(result, indent=2, default=str))