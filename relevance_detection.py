
# relevance_detection.py - FINAL FIXED VERSION
"""
Milestone 5: Relevance Detection & LangGraph Integration - FIXED
- Compare news embeddings to company graph entities
- Route events through processing steps using LangGraph
- Trigger research agent for high-relevance matches
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# LangChain and LangGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Import our modules
from news_pipeline import NewsArticle, NewsDatabase, NewsProcessor

from dotenv import load_dotenv
load_dotenv()

@dataclass
class RelevanceMatch:
    """Represents a relevance match between news and knowledge graph entities"""
    news_article_id: str
    news_title: str
    news_url: str
    news_content_preview: str
    
    matched_entities: List[Dict[str, Any]]
    graph_entities: List[str]
    
    relevance_score: float
    relevance_category: str  # HIGH, MEDIUM, LOW
    
    # Analysis details
    entity_matches: List[Dict[str, Any]]
    semantic_similarity: float
    keyword_matches: List[str]
    
    # Processing metadata
    created_at: datetime
    processed_by_agent: bool = False
    agent_analysis: Optional[str] = None

class EntityMatcher:
    """Matches news entities with knowledge graph entities"""
    
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.entity_embeddings_cache = {}
    
    def extract_graph_entities(self, G: nx.DiGraph) -> Dict[str, Dict]:
        """Extract entities from knowledge graph with metadata"""
        entities = {}
        
        for node_id, attrs in G.nodes(data=True):
            # Skip category nodes and root
            if (node_id.startswith("cat::") or 
                attrs.get("type") in ["Category", "Subcategory"] or
                node_id == "apple"):
                continue
            
            entities[node_id] = {
                "name": attrs.get("name", node_id),
                "type": attrs.get("type", "Unknown"),
                "description": attrs.get("prop_description", ""),
                "risk_level": attrs.get("prop_risk_level", "Unknown")
            }
        
        return entities
    
    def get_entity_embedding(self, entity_name: str, entity_description: str = "") -> List[float]:
        """Get or create embedding for an entity"""
        cache_key = f"{entity_name}:{entity_description}"
        
        if cache_key in self.entity_embeddings_cache:
            return self.entity_embeddings_cache[cache_key]
        
        # Create text for embedding
        text = entity_name
        if entity_description:
            text += f" - {entity_description}"
        
        try:
            embedding = self.embeddings.embed_query(text)
            self.entity_embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error creating embedding for entity {entity_name}: {e}")
            return []
    
    def calculate_semantic_similarity(
        self, 
        news_embedding: List[float], 
        entity_embedding: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            if not news_embedding or not entity_embedding:
                return 0.0
            
            # Ensure embeddings are numpy arrays and properly shaped
            news_emb = np.array(news_embedding).reshape(1, -1)
            entity_emb = np.array(entity_embedding).reshape(1, -1)
            
            # Normalize embeddings to ensure cosine similarity is in [-1, 1]
            news_emb = news_emb / np.linalg.norm(news_emb)
            entity_emb = entity_emb / np.linalg.norm(entity_emb)
            
            similarity = cosine_similarity(news_emb, entity_emb)[0][0]
            
            # Ensure similarity is in valid range
            similarity = max(-1.0, min(1.0, float(similarity)))
            
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_keyword_matches(
        self, 
        news_text: str, 
        entity_name: str, 
        entity_description: str = ""
    ) -> List[str]:
        """Find direct keyword matches between news and entity"""
        matches = []
        news_lower = news_text.lower()
        
        # Check entity name (exact match)
        if entity_name.lower() in news_lower:
            matches.append(entity_name)
        
        # Check words in entity name (individual words, only if meaningful)
        entity_words = entity_name.lower().split()
        for word in entity_words:
            if len(word) > 4 and word in news_lower:  # Increased minimum length
                matches.append(word)
        
        # Check description keywords (only significant words)
        if entity_description:
            desc_words = entity_description.lower().split()
            for word in desc_words:
                if len(word) > 5 and word in news_lower:  # Even more restrictive
                    matches.append(word)
        
        return list(set(matches))  # Remove duplicates
    
    def match_article_to_entities(
        self, 
        article: NewsArticle, 
        graph_entities: Dict[str, Dict],
        similarity_threshold: float = 0.7,  # Much higher threshold
        max_matches: int = 10  # Limit number of matches
    ) -> List[Dict[str, Any]]:
        """Match a news article to relevant graph entities"""
        
        if not article.embedding:
            print(f"  Warning: Article has no embedding")
            return []
        
        matches = []
        full_text = f"{article.title} {article.content}"
        
        print(f"  Analyzing text for Apple-related entities...")
        
        # Pre-filter entities that might be relevant
        apple_keywords = ['apple', 'iphone', 'ipad', 'mac', 'ios', 'tim cook', 'foxconn', 'tsmc']
        potentially_relevant = []
        
        for entity_id, entity_data in graph_entities.items():
            entity_name_lower = entity_data["name"].lower()
            entity_desc_lower = entity_data["description"].lower()
            
            # Check if any Apple-related keywords appear in the text
            has_apple_context = any(keyword in full_text.lower() for keyword in apple_keywords)
            
            # Check if entity is mentioned or related
            entity_mentioned = (entity_name_lower in full_text.lower() or
                              any(word in full_text.lower() for word in entity_name_lower.split() if len(word) > 4))
            
            # Only consider entities that are either mentioned or in Apple context
            if entity_mentioned or (has_apple_context and entity_data["type"] in ["Supplier", "Product", "Component"]):
                potentially_relevant.append((entity_id, entity_data))
        
        print(f"  Found {len(potentially_relevant)} potentially relevant entities")
        
        if not potentially_relevant:
            return []
        
        for entity_id, entity_data in potentially_relevant:
            # Get entity embedding
            entity_embedding = self.get_entity_embedding(
                entity_data["name"],
                entity_data["description"]
            )
            
            if not entity_embedding:
                continue
            
            # Calculate semantic similarity
            semantic_sim = self.calculate_semantic_similarity(
                article.embedding,
                entity_embedding
            )
            
            # Find keyword matches
            keyword_matches = self.find_keyword_matches(
                full_text,
                entity_data["name"],
                entity_data["description"]
            )
            
            # Calculate combined relevance score with more conservative weights
            keyword_score = len(keyword_matches) * 0.1  # Reduced from 0.2
            combined_score = semantic_sim + keyword_score
            
            # Check if entity was mentioned in extracted entities
            entity_mention_bonus = 0.0
            if article.entities:
                for extracted_entity in article.entities:
                    extracted_text = extracted_entity.get("text", "").lower()
                    entity_name_lower = entity_data["name"].lower()
                    
                    if (entity_name_lower in extracted_text or
                        any(word in extracted_text for word in entity_name_lower.split() if len(word) > 4)):
                        entity_mention_bonus = 0.2  # Reduced from 0.3
                        break
            
            final_score = combined_score + entity_mention_bonus
            
            # More strict threshold - only include meaningful matches
            if final_score >= similarity_threshold or len(keyword_matches) >= 2:  # Need 2+ keywords
                matches.append({
                    "entity_id": entity_id,
                    "entity_name": entity_data["name"],
                    "entity_type": entity_data["type"],
                    "semantic_similarity": semantic_sim,
                    "keyword_matches": keyword_matches,
                    "entity_mention_bonus": entity_mention_bonus,
                    "combined_score": final_score,
                    "risk_level": entity_data.get("risk_level", "Unknown")
                })
        
        # Sort by combined score and limit results
        matches.sort(key=lambda x: x["combined_score"], reverse=True)
        matches = matches[:max_matches]
        
        if matches:
            print(f"  Found {len(matches)} high-quality entity matches (best: {matches[0]['combined_score']:.3f})")
        else:
            print(f"  No high-quality entity matches found")
        
        return matches

class RelevanceScorer:
    """Determines overall relevance of news to Apple's business"""
    
    def __init__(self):
        # More conservative scoring weights
        self.weights = {
            "high_impact_entities": 2.0,    # Reduced from 4.0
            "medium_impact_entities": 1.5,  # Reduced from 2.5
            "low_impact_entities": 1.0,     # Reduced from 1.5
            "sentiment_multiplier": 1.3,    # Reduced from 1.8
            "recency_multiplier": 1.2,      # Reduced from 1.4
            "source_credibility": 1.2       # Reduced from 1.5
        }
        
        # High-impact entity types
        self.high_impact_types = ["Supplier", "Product", "Component"]
        self.medium_impact_types = ["Location", "Material", "Service"]
        
        # Trusted news sources
        self.trusted_sources = [
            "reuters", "bloomberg", "financial times", "wall street journal",
            "associated press", "bbc", "techcrunch", "ars technica", "macrumors", "9to5mac"
        ]
    
    def calculate_entity_impact_score(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate impact score based on matched entities"""
        if not matches:
            return 0.0
        
        total_score = 0.0
        
        for match in matches:
            entity_type = match.get("entity_type", "").lower()
            base_score = match.get("combined_score", 0.0)
            
            # Apply type-based multiplier
            if any(impact_type.lower() in entity_type for impact_type in self.high_impact_types):
                multiplier = self.weights["high_impact_entities"]
            elif any(impact_type.lower() in entity_type for impact_type in self.medium_impact_types):
                multiplier = self.weights["medium_impact_entities"]
            else:
                multiplier = self.weights["low_impact_entities"]
            
            # Risk level bonus (more conservative)
            risk_level = match.get("risk_level", "").lower()
            risk_multiplier = 1.3 if risk_level == "high" else 1.1 if risk_level == "medium" else 1.0
            
            total_score += base_score * multiplier * risk_multiplier
        
        # More realistic cap
        return min(total_score, 5.0)  # Reduced from 10.0
    
    def calculate_sentiment_multiplier(self, article: NewsArticle) -> float:
        """Calculate multiplier based on article sentiment"""
        if not article.sentiment_score:
            return 1.0
        
        # Negative news often more impactful for risk monitoring
        if article.sentiment_score < -0.3:
            return self.weights["sentiment_multiplier"]
        elif article.sentiment_score < -0.1:
            return 1.15
        else:
            return 1.0
    
    def calculate_recency_multiplier(self, article: NewsArticle) -> float:
        """Calculate multiplier based on article recency"""
        hours_old = (datetime.now() - article.published_date).total_seconds() / 3600
        
        if hours_old < 6:
            return self.weights["recency_multiplier"]
        elif hours_old < 24:
            return 1.1
        else:
            return 1.0
    
    def calculate_source_multiplier(self, article: NewsArticle) -> float:
        """Calculate multiplier based on source credibility"""
        source_lower = article.source.lower()
        
        for trusted_source in self.trusted_sources:
            if trusted_source in source_lower:
                return self.weights["source_credibility"]
        
        return 1.0
    
    def calculate_relevance_score(
        self, 
        article: NewsArticle, 
        entity_matches: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Calculate overall relevance score and category"""
        
        # Base score from entity matches
        entity_score = self.calculate_entity_impact_score(entity_matches)
        
        # Apply multipliers only if there's a meaningful base score
        if entity_score > 0.5:  # Minimum threshold for applying multipliers
            sentiment_mult = self.calculate_sentiment_multiplier(article)
            recency_mult = self.calculate_recency_multiplier(article)
            source_mult = self.calculate_source_multiplier(article)
            
            final_score = entity_score * sentiment_mult * recency_mult * source_mult
        else:
            final_score = entity_score  # Don't amplify very low scores
        
        # More realistic category thresholds
        if final_score >= 4.0:
            category = "HIGH"
        elif final_score >= 2.0:
            category = "MEDIUM"
        elif final_score >= 0.5:
            category = "LOW"
        else:
            category = "MINIMAL"
        
        return min(final_score, 10.0), category

# LangGraph State and Workflow
class RelevanceState(TypedDict):
    """State for relevance detection workflow"""
    article: NewsArticle
    graph_entities: Dict[str, Dict]
    entity_matches: List[Dict[str, Any]]
    relevance_score: float
    relevance_category: str
    requires_research: bool
    messages: Annotated[List[BaseMessage], add_messages]
    analysis_complete: bool

class RelevanceWorkflow:
    """LangGraph workflow for processing news relevance"""
    
    def __init__(self, openai_api_key: str):
        self.matcher = EntityMatcher(openai_api_key)
        self.scorer = RelevanceScorer()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(RelevanceState)
        
        # Add nodes
        workflow.add_node("match_entities", self.match_entities_node)
        workflow.add_node("score_relevance", self.score_relevance_node)
        workflow.add_node("analyze_impact", self.analyze_impact_node)
        workflow.add_node("trigger_research", self.trigger_research_node)
        
        # Define the flow
        workflow.set_entry_point("match_entities")
        
        workflow.add_edge("match_entities", "score_relevance")
        workflow.add_edge("score_relevance", "analyze_impact")
        
        # Conditional routing based on relevance score
        workflow.add_conditional_edges(
            "analyze_impact",
            self.should_trigger_research,
            {
                "research": "trigger_research",
                "end": END
            }
        )
        
        workflow.add_edge("trigger_research", END)
        
        return workflow.compile()
    
    def match_entities_node(self, state: RelevanceState) -> RelevanceState:
        """Node: Match article entities with graph entities"""
        
        article = state["article"]
        graph_entities = state["graph_entities"]
        
        # Perform entity matching with stricter thresholds
        entity_matches = self.matcher.match_article_to_entities(
            article, 
            graph_entities,
            similarity_threshold=0.7,
            max_matches=5
        )
        
        state["entity_matches"] = entity_matches
        state["messages"].append(
            HumanMessage(content=f"Found {len(entity_matches)} entity matches for article: {article.title}")
        )
        
        return state
    
    def score_relevance_node(self, state: RelevanceState) -> RelevanceState:
        """Node: Calculate relevance score"""
        
        article = state["article"]
        entity_matches = state["entity_matches"]
        
        # Calculate relevance
        relevance_score, relevance_category = self.scorer.calculate_relevance_score(
            article, entity_matches
        )
        
        state["relevance_score"] = relevance_score
        state["relevance_category"] = relevance_category
        
        state["messages"].append(
            HumanMessage(content=f"Relevance score: {relevance_score:.2f} ({relevance_category})")
        )
        
        return state
    
    def analyze_impact_node(self, state: RelevanceState) -> RelevanceState:
        """Node: Analyze potential business impact using LLM"""
        
        article = state["article"]
        entity_matches = state["entity_matches"]
        relevance_score = state["relevance_score"]
        
        # Only do detailed analysis if relevance score is meaningful
        if relevance_score < 2.0:  # Increased threshold
            state["requires_research"] = False
            return state
        
        # Create analysis prompt
        matched_entities_text = ", ".join([
            f"{match['entity_name']} ({match['entity_type']})" 
            for match in entity_matches[:3]  # Only top 3
        ])
        
        prompt_template = """
        Analyze the potential business impact of this news article on Apple Inc.
        
        Article Title: {title}
        Article Content: {content}
        
        Matched Apple-related Entities: {entities}
        Relevance Score: {score}
        
        Consider ONLY direct impacts:
        1. Supply chain disruptions affecting Apple
        2. Regulatory changes affecting Apple's products/services
        3. Competitive threats or opportunities for Apple
        4. Financial/market impacts specific to Apple
        
        Provide a brief analysis (1-2 sentences) focusing ONLY on direct Apple impacts.
        Rate the urgency as: IMMEDIATE, HIGH, MEDIUM, or LOW.
        
        If there is no clear direct impact on Apple, respond with "No direct Apple impact identified."
        
        Format:
        Impact Analysis: [your analysis or "No direct Apple impact identified"]
        Urgency: [urgency level or "N/A"]
        """
        
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm
            
            result = chain.invoke({
                "title": article.title,
                "content": article.content[:800],  # Reduced content length
                "entities": matched_entities_text,
                "score": relevance_score
            }).content
            
            state["messages"].append(
                SystemMessage(content=f"LLM Impact Analysis: {result}")
            )
            
            # More strict research trigger conditions
            urgency_levels = ["IMMEDIATE", "HIGH"]
            has_direct_impact = "No direct Apple impact identified" not in result
            has_high_urgency = any(level in result.upper() for level in urgency_levels)
            
            state["requires_research"] = (has_direct_impact and 
                                        has_high_urgency and 
                                        relevance_score >= 3.0)  # Even higher threshold
            
        except Exception as e:
            print(f"Error in impact analysis: {e}")
            state["requires_research"] = False
        
        return state
    
    def should_trigger_research(self, state: RelevanceState) -> str:
        """Conditional edge: Determine if research agent should be triggered"""
        
        # Very strict research trigger conditions
        score = state["relevance_score"]
        category = state["relevance_category"]
        requires_research = state.get("requires_research", False)
        
        # Only trigger for high scores AND explicit requirement AND high category
        if (score >= 4.0 and category == "HIGH" and requires_research):
            return "research"
        else:
            return "end"
    
    def trigger_research_node(self, state: RelevanceState) -> RelevanceState:
        """Node: Trigger research agent for high-priority articles"""
        
        article = state["article"]
        
        research_trigger = {
            "article_id": article.id,
            "article_title": article.title,
            "article_url": article.url,
            "relevance_score": state["relevance_score"],
            "matched_entities": [match["entity_name"] for match in state["entity_matches"][:3]],
            "triggered_at": datetime.now().isoformat()
        }
        
        print(f"🔥 RESEARCH TRIGGERED for: {article.title}")
        print(f"   Score: {state['relevance_score']:.2f}")
        print(f"   Entities: {', '.join(research_trigger['matched_entities'])}")
        
        state["messages"].append(
            SystemMessage(content=f"Research agent triggered for high-priority article: {article.title}")
        )
        
        state["analysis_complete"] = True
        
        return state
    
    def process_article(
        self, 
        article: NewsArticle, 
        graph_entities: Dict[str, Dict]
    ) -> RelevanceMatch:
        """Process a single article through the workflow"""
        
        # Initialize state
        initial_state = RelevanceState(
            article=article,
            graph_entities=graph_entities,
            entity_matches=[],
            relevance_score=0.0,
            relevance_category="LOW",
            requires_research=False,
            messages=[],
            analysis_complete=False
        )
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Extract keyword matches from entity matches
            all_keyword_matches = []
            for match in final_state["entity_matches"]:
                all_keyword_matches.extend(match.get("keyword_matches", []))
            
            # Create relevance match object
            match = RelevanceMatch(
                news_article_id=article.id,
                news_title=article.title,
                news_url=article.url,
                news_content_preview=article.content[:200] + "..." if len(article.content) > 200 else article.content,
                matched_entities=article.entities or [],
                graph_entities=[match["entity_name"] for match in final_state["entity_matches"]],
                relevance_score=final_state["relevance_score"],
                relevance_category=final_state["relevance_category"],
                entity_matches=final_state["entity_matches"],
                semantic_similarity=np.mean([match["semantic_similarity"] for match in final_state["entity_matches"]]) if final_state["entity_matches"] else 0.0,
                keyword_matches=all_keyword_matches,
                created_at=datetime.now(),
                processed_by_agent=final_state.get("requires_research", False)
            )
            
            return match
            
        except Exception as e:
            print(f"Error processing article through workflow: {e}")
            # Return minimal match object
            return RelevanceMatch(
                news_article_id=article.id,
                news_title=article.title,
                news_url=article.url,
                news_content_preview=article.content[:200],
                matched_entities=[],
                graph_entities=[],
                relevance_score=0.0,
                relevance_category="ERROR",
                entity_matches=[],
                semantic_similarity=0.0,
                keyword_matches=[],
                created_at=datetime.now()
            )

class RelevanceMonitor:
    """Main orchestrator for relevance detection and monitoring"""
    
    def __init__(
        self, 
        openai_api_key: str,
        news_db_path: str = "./apple_news_db",
        relevance_db_path: str = "./relevance_matches.json"
    ):
        self.workflow = RelevanceWorkflow(openai_api_key)
        self.news_db = NewsDatabase(news_db_path)
        self.relevance_db_path = relevance_db_path
        
        # Load existing relevance matches
        self.relevance_matches = self._load_relevance_matches()
    
    def _load_relevance_matches(self) -> List[RelevanceMatch]:
        """Load existing relevance matches from disk - FIXED datetime handling"""
        try:
            if os.path.exists(self.relevance_db_path):
                with open(self.relevance_db_path, 'r') as f:
                    data = json.load(f)
                    
                    matches = []
                    for item in data:
                        # Fix datetime field if it's a string
                        if isinstance(item.get('created_at'), str):
                            try:
                                item['created_at'] = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                            except:
                                item['created_at'] = datetime.now()
                        
                        matches.append(RelevanceMatch(**item))
                    
                    return matches
        except Exception as e:
            print(f"Error loading relevance matches: {e}")
        return []
    
    def _save_relevance_matches(self):
        """Save relevance matches to disk"""
        try:
            data = [asdict(match) for match in self.relevance_matches]
            # Convert datetime objects to strings
            for item in data:
                item["created_at"] = item["created_at"].isoformat() if isinstance(item["created_at"], datetime) else item["created_at"]
            
            with open(self.relevance_db_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving relevance matches: {e}")
    
    def process_recent_news(
        self, 
        knowledge_graph: nx.DiGraph,
        days_back: int = 1,
        min_relevance_score: float = 1.5  # Increased minimum
    ) -> List[RelevanceMatch]:
        """Process recent news articles for relevance"""
        
        print(f"=== Processing news from last {days_back} days ===")
        
        # Extract graph entities
        matcher = EntityMatcher(self.workflow.llm.openai_api_key)
        graph_entities = matcher.extract_graph_entities(knowledge_graph)
        print(f"Found {len(graph_entities)} entities in knowledge graph")
        
        # Get recent articles from database
        recent_results = self.news_db.get_recent_articles(days=days_back)
        
        if not recent_results["documents"]:
            print("No recent articles found")
            return []
        
        print(f"Processing {len(recent_results['documents'])} recent articles...")
        
        new_matches = []
        processed_article_ids = {match.news_article_id for match in self.relevance_matches}
        
        for i, (doc, metadata) in enumerate(zip(recent_results["documents"], recent_results["metadatas"])):
            try:
                # Skip if already processed
                article_id = recent_results["ids"][i] if "ids" in recent_results else f"unknown_{i}"
                
                if article_id in processed_article_ids:
                    continue
                
                # Reconstruct article object
                try:
                    published_date = datetime.fromisoformat(metadata.get("published_date", datetime.now().isoformat()))
                except:
                    published_date = datetime.now()
                
                article = NewsArticle(
                    id=article_id,
                    title=metadata.get("title", ""),
                    content=doc,
                    url=metadata.get("url", ""),
                    source=metadata.get("source", ""),
                    published_date=published_date,
                    author=metadata.get("author"),
                    cleaned_content=doc,
                    entities=json.loads(metadata.get("entities", "[]")),
                    sentiment_score=metadata.get("sentiment_score", 0.0),
                    sentiment_label=metadata.get("sentiment_label", "neutral"),
                    relevance_score=metadata.get("relevance_score", 0.0)
                )
                
                # Create embedding if missing
                if not hasattr(article, 'embedding') or not article.embedding:
                    try:
                        embeddings = OpenAIEmbeddings(openai_api_key=self.workflow.llm.openai_api_key)
                        article.embedding = embeddings.embed_query(article.cleaned_content)
                    except Exception as e:
                        print(f"  Error creating embedding: {e}")
                        continue
                
                # Process through workflow
                print(f"Processing article {i+1}: {article.title[:50]}...")
                match = self.workflow.process_article(article, graph_entities)
                
                # Only keep if above minimum relevance
                if match.relevance_score >= min_relevance_score:
                    new_matches.append(match)
                    self.relevance_matches.append(match)
                    
                    print(f"  ✅ High relevance: {match.relevance_score:.2f} ({match.relevance_category})")
                else:
                    print(f"  ⚪ Low relevance: {match.relevance_score:.2f}")
                
            except Exception as e:
                print(f"Error processing article {i}: {e}")
                continue
        
        # Save updated matches
        self._save_relevance_matches()
        
        print(f"=== Processing complete: {len(new_matches)} new high-relevance matches ===")
        return new_matches
    
    def get_top_matches(self, limit: int = 10) -> List[RelevanceMatch]:
        """Get top relevance matches by score"""
        sorted_matches = sorted(
            self.relevance_matches, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        return sorted_matches[:limit]
    
    def get_matches_by_category(self, category: str) -> List[RelevanceMatch]:
        """Get matches by relevance category"""
        return [
            match for match in self.relevance_matches 
            if match.relevance_category == category.upper()
        ]

# Example usage and integration
if __name__ == "__main__":
    import sys
    sys.path.append(".")  # Add current directory to path
    
    from knowledge_graph import load_documents, split_documents, process_all_chunks, create_networkx_graph, ensure_hierarchy
    
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Load knowledge graph (assuming it exists)
    try:
        import pickle
        with open("apple_knowledge_graph.gpickle", 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded knowledge graph with {G.number_of_nodes()} nodes")
    except FileNotFoundError:
        print("Knowledge graph not found. Please run knowledge_graph.py first.")
        exit(1)
    
    # Initialize relevance monitor
    monitor = RelevanceMonitor(
        openai_api_key=OPENAI_API_KEY,
        news_db_path="./apple_news_db"
    )
    
    # Process recent news
    matches = monitor.process_recent_news(G, days_back=7, min_relevance_score=1.5)
    
    # Display results
    print("\n=== TOP RELEVANCE MATCHES ===")
    top_matches = monitor.get_top_matches(5)
    
    for i, match in enumerate(top_matches, 1):
        print(f"\n{i}. {match.news_title}")
        print(f"   Score: {match.relevance_score:.2f} ({match.relevance_category})")
        print(f"   URL: {match.news_url}")
        print(f"   Entities: {', '.join(match.graph_entities[:3])}")
        print(f"   Research Triggered: {'Yes' if match.processed_by_agent else 'No'}")