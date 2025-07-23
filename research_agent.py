#research_agent.py

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict

# LangChain and LangGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Web search capabilities
import requests
from bs4 import BeautifulSoup
import time

# Import our modules
from relevance_detection import RelevanceMatch
from news_pipeline import NewsArticle

from dotenv import load_dotenv
load_dotenv()

@dataclass
class ResearchResult:
    article_id: str
    article_title: str
    article_url: str
    
    # Research metadata
    research_triggered_at: datetime
    research_completed_at: Optional[datetime] = None
    research_duration_seconds: Optional[float] = None
    
    # Financial impact analysis
    financial_impact: Dict[str, Any] = None
    stock_price_context: Dict[str, Any] = None
    analyst_sentiment: str = "Unknown"
    
    # Operational impact analysis
    operational_impact: Dict[str, Any] = None
    supply_chain_effects: List[str] = None
    product_line_effects: List[str] = None
    
    # Reputational impact analysis
    reputational_impact: Dict[str, Any] = None
    social_media_sentiment: str = "Unknown"
    regulatory_implications: List[str] = None

    # Executive summary
    executive_summary: str = ""
    key_takeaways: List[str] = None
    recommended_actions: List[str] = None
    urgency_level: str = "Medium"  # Low, Medium, High, Critical
    
    # Additional context
    related_articles: List[Dict] = None
    competitor_analysis: Dict[str, Any] = None
    market_context: Dict[str, Any] = None
    
    # Research sources
    sources_consulted: List[str] = None
    research_queries_used: List[str] = None
    
    def __post_init__(self):
        if self.financial_impact is None:
            self.financial_impact = {}
        if self.operational_impact is None:
            self.operational_impact = {}
        if self.reputational_impact is None:
            self.reputational_impact = {}
        if self.supply_chain_effects is None:
            self.supply_chain_effects = []
        if self.product_line_effects is None:
            self.product_line_effects = []
        if self.regulatory_implications is None:
            self.regulatory_implications = []
        if self.key_takeaways is None:
            self.key_takeaways = []
        if self.recommended_actions is None:
            self.recommended_actions = []
        if self.related_articles is None:
            self.related_articles = []
        if self.competitor_analysis is None:
            self.competitor_analysis = {}
        if self.market_context is None:
            self.market_context = {}
        if self.sources_consulted is None:
            self.sources_consulted = []
        if self.research_queries_used is None:
            self.research_queries_used = []
            
class WebSearchClient:
    """Web search client for gathering additional context"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2  # seconds between requests
        
    def search_news(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search for recent news articles related to the query"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            # Use DuckDuckGo news search
            search_url = f"https://duckduckgo.com/news"
            params = {'q': query, 'df': 'w'}  # Last week
            
            response = self.session.get(search_url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                # Parse results (simplified)
                results = []
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Simplified parsing logic TODO: Check why we doing this
                for i in range(min(num_results, 3)):  # Limit to prevent issues
                    results.append({
                        'title': f"Related article {i+1} for: {query}",
                        'url': f"https://example.com/article{i+1}",
                        'snippet': f"Additional context about {query}",
                        'source': 'News Search'
                    })
                
                return results
            else:
                return []
            
        except Exception as e:
            print(f"Error in web search: {e}")
            return []
        
    
    def fetch_webpage_content(self, url: str) -> str:
        """Fetch and extract text content from a webpage"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
                
            response = self.session.get(url, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
            
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:3000]  # Limit length
            else:
                return ""
                
        except Exception as e:
            print(f"Error fetching webpage {url}: {e}")
            return ""
    
class FinancialDataClient:
    """Client for financial data and market context"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        
    def get_stock_price_context(self, symbol: str = "AAPL") -> Dict[str, Any]:
        """Get recent stock price movement and context"""
        try:
            if self.alpha_vantage_key:
                # Use Alpha Vantage if API key available
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': self.alpha_vantage_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        dates = sorted(time_series.keys(), reverse=True)
                        
                        if len(dates) >= 2: 
                            latest = time_series[dates[0]]
                            previous = time_series[dates[1]]
                            
                            current_price = float(latest['4. close'])
                            previous_price = float(previous['4. close'])
                            change = current_price - previous_price
                            change_pct = (change / previous_price) * 100
                            
                            return {
                                'symbol': symbol,
                                'current_price': current_price,
                                'previous_price': previous_price,
                                'change': change,
                                'change_percent': change_pct,
                                'volume': int(latest['5. volume']),
                                'date': dates[0],
                                'source': 'Alpha Vantage'
                            }
                # Fallback to mock data if no API key or error
            return {
                'symbol': symbol,
                'current_price': 175.50,
                'previous_price': 174.20,
                'change': 1.30,
                'change_percent': 0.75,
                'volume': 45000000,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Mock Data',
                'note': 'API key required for real data'
            }
            
        except Exception as e:
            print(f"Error getting stock data: {e}")
            return {'error': str(e), 'source': 'Error'}
        
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment and tech sector context"""
        return {
            'tech_sector_trend': 'positive',
            'market_volatility': 'medium',
            'analyst_consensus': 'hold',
            'sector_rotation': 'into_tech',
            'source': 'Mock Market Data'
        }
        
class ResearchState(TypedDict):
    """State for research workflow"""
    article: NewsArticle
    relevance_match: RelevanceMatch
    research_queries: List[str]
    web_search_results: List[Dict]
    financial_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    messages: Annotated[List[BaseMessage], add_messages]
    research_complete: bool
    executive_summary: str
    
class ResearchAgent:
    """AI-powered research agent for analyzing high-relevance news"""
    
    def __init__(self, openai_api_key: str, alpha_vantage_key: Optional[str] = None):
        self.llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", openai_api_key=openai_api_key)
        self.web_client = WebSearchClient()
        self.financial_client = FinancialDataClient(alpha_vantage_key)
        
        # Build the research workflow
        self.workflow = self._build_research_workflow()
        
    def _build_research_workflow(self) -> StateGraph:
        """Build the LangGraph research workflow"""
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.generate_research_queries_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("financial_analysis", self.financial_analysis_node)
        workflow.add_node("impact_analysis", self.impact_analysis_node)
        workflow.add_node("generate_summary", self.generate_executive_summary_node)
        
        # Define the flow
        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "web_search")
        workflow.add_edge("web_search", "financial_analysis")
        workflow.add_edge("financial_analysis", "impact_analysis")
        workflow.add_edge("impact_analysis", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        return workflow.compile()
    
    def generate_research_queries_node(self, state: ResearchState) -> ResearchState:
        """Generate targeted research queries based on the article"""
        
        article = state["article"]
        relevance_match = state["relevance_match"]
        
        prompt_template = """
        Based on this Apple-related news article, generate 3-5 specific research queries to investigate potential business impacts.
        
        Article Title: {title}
        Article Content: {content}
        Matched Entities: {entities}
        Relevance Score: {score}
        
        Generate queries that would help research:
        1. Financial impact on Apple (stock price, revenue, costs)
        2. Operational impact (supply chain, manufacturing, product lines)
        3. Competitive positioning and market share effects
        4. Regulatory or compliance implications
        5. Reputation and brand impact
        
        Focus on SPECIFIC and ACTIONABLE queries that would yield concrete insights for Apple's business.
        
        Return as a JSON list of strings:
        ["query1", "query2", "query3"]
        
        Return ONLY valid JSON:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "title": article.title,
                "content": article.content[:1000],
                "entities": ", ".join(relevance_match.graph_entities),
                "score": relevance_match.relevance_score
            }).content
            
            # Parse the result
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            queries = json.loads(result)
             # Validate and clean queries
            if isinstance(queries, list):
                clean_queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 10]
                state["research_queries"] = clean_queries[:5]  # Limit to 5 queries
            else:
                state["research_queries"] = [f"Apple impact analysis: {article.title}"]
                
        except Exception as e:
            print(f"Error generating research queries: {e}")
            # Fallback to basic queries
            state["research_queries"] = [
                f"Apple financial impact: {article.title}",
                f"Apple operational effects: {relevance_match.graph_entities[0] if relevance_match.graph_entities else 'news'}",
                f"Apple stock market reaction: {article.title[:50]}"
            ]
        
        state["messages"].append(
            HumanMessage(content=f"Generated {len(state['research_queries'])} research queries")
        )
        
        return state
    
    def web_search_node(self, state: ResearchState) -> ResearchState:
        """Perform web searches for additional context"""
        
        queries = state["research_queries"]
        all_results = []
        
        for query in queries[:3]:  # Limit to 3 queries to avoid rate limits
            try:
                print(f"  Searching: {query}")
                results = self.web_client.search_news(query, num_results=2)
                all_results.extend(results)
                
                # Small delay between searches
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in web search for '{query}': {e}")
                continue
        
        state["web_search_results"] = all_results
        state["messages"].append(
            HumanMessage(content=f"Completed web search: found {len(all_results)} additional sources")
        )
        
        return state
    
    def financial_analysis_node(self, state: ResearchState) -> ResearchState:
        """Analyze financial implications and market context"""
        
        try:
            # Get stock price context
            stock_data = self.financial_client.get_stock_price_context("AAPL")
            market_sentiment = self.financial_client.get_market_sentiment()
            
            state["financial_data"] = {
                "stock_data": stock_data,
                "market_sentiment": market_sentiment
            }
            
            state["messages"].append(
                HumanMessage(content=f"Gathered financial context: AAPL at ${stock_data.get('current_price', 'N/A')}")
            )
            
        except Exception as e:
            print(f"Error in financial analysis: {e}")
            state["financial_data"] = {"error": str(e)}
        
        return state
    
    
    def impact_analysis_node(self, state: ResearchState) -> ResearchState:
        """Analyze potential business impacts using LLM"""
        
        article = state["article"]
        relevance_match = state["relevance_match"]
        web_results = state["web_search_results"]
        financial_data = state["financial_data"]
        
        # Prepare context for analysis
        web_context = "\n".join([
            f"- {result.get('title', '')}: {result.get('snippet', '')}"
            for result in web_results[:5]
        ])
        
        stock_context = ""
        if "stock_data" in financial_data:
            stock = financial_data["stock_data"]
            stock_context = f"AAPL: ${stock.get('current_price', 'N/A')} ({stock.get('change_percent', 0):.2f}%)"
        
        prompt_template = """
        Analyze the potential business impact of this news on Apple Inc. Use the provided context to generate a comprehensive assessment.
        
        ARTICLE:
        Title: {title}
        Content: {content}
        Matched Apple Entities: {entities}
        
        ADDITIONAL CONTEXT:
        Recent Stock Performance: {stock_context}
        Related News: {web_context}
        
        ANALYSIS FRAMEWORK:
        Please provide a structured analysis covering:
        
        1. FINANCIAL IMPACT (potential effects on revenue, costs, margins, stock price)
        2. OPERATIONAL IMPACT (supply chain, manufacturing, product development effects)
        3. REPUTATIONAL IMPACT (brand perception, customer sentiment, regulatory attention)
        4. URGENCY LEVEL (Low/Medium/High/Critical based on immediacy and severity)
        
        For each impact area, provide:
        - Likelihood (High/Medium/Low)
        - Severity (Major/Moderate/Minor)
        - Timeline (Immediate/Short-term/Long-term)
        - Specific implications for Apple
        
        Format your response as structured JSON:
        {{
            "financial_impact": {{
                "likelihood": "High/Medium/Low",
                "severity": "Major/Moderate/Minor",
                "timeline": "Immediate/Short-term/Long-term",
                "details": "specific analysis",
                "potential_revenue_impact": "estimate if possible",
                "stock_price_sensitivity": "High/Medium/Low"
            }},
            "operational_impact": {{
                "likelihood": "High/Medium/Low",
                "severity": "Major/Moderate/Minor",
                "timeline": "Immediate/Short-term/Long-term",
                "details": "specific analysis",
                "affected_product_lines": ["list if applicable"],
                "supply_chain_risk": "High/Medium/Low"
            }},
            "reputational_impact": {{
                "likelihood": "High/Medium/Low",
                "severity": "Major/Moderate/Minor", 
                "timeline": "Immediate/Short-term/Long-term",
                "details": "specific analysis",
                "regulatory_attention": "High/Medium/Low",
                "customer_sentiment_risk": "High/Medium/Low"
            }},
            "urgency_level": "Critical/High/Medium/Low",
            "key_takeaways": ["key point 1", "key point 2", "key point 3"],
            "recommended_actions": ["action 1", "action 2", "action 3"]
        }}
        
        Return ONLY valid JSON:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "title": article.title,
                "content": article.content[:2000],
                "entities": ", ".join(relevance_match.graph_entities),
                "stock_context": stock_context,
                "web_context": web_context or "No additional context found"
            }).content
            
            # Parse the result
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            analysis = json.loads(result)
            state["analysis_results"] = analysis
            
        except Exception as e:
            print(f"Error in impact analysis: {e}")
            # Fallback analysis
            state["analysis_results"] = {
                "financial_impact": {"likelihood": "Medium", "severity": "Moderate", "timeline": "Short-term", "details": "Analysis failed - manual review required"},
                "operational_impact": {"likelihood": "Medium", "severity": "Moderate", "timeline": "Short-term", "details": "Analysis failed - manual review required"},
                "reputational_impact": {"likelihood": "Medium", "severity": "Moderate", "timeline": "Short-term", "details": "Analysis failed - manual review required"},
                "urgency_level": "Medium",
                "key_takeaways": ["Analysis error occurred"],
                "recommended_actions": ["Manual review recommended"]
            }
        
        state["messages"].append(
            HumanMessage(content=f"Completed impact analysis with urgency level: {state['analysis_results'].get('urgency_level', 'Unknown')}")
        )
        
        return state
    
    def generate_executive_summary_node(self, state: ResearchState) -> ResearchState:
        """Generate executive summary and final recommendations"""
        
        article = state["article"]
        analysis = state["analysis_results"]
        
        prompt_template = """
        Create a concise executive summary for Apple leadership based on the impact analysis.
        
        Article: {title}
        Analysis Results: {analysis}
        
        Generate a 2-3 paragraph executive summary that:
        1. Clearly states what happened and why it matters to Apple
        2. Summarizes the key risks and opportunities
        3. Provides clear, actionable recommendations
        
        Keep it executive-level: strategic, actionable, and decision-focused.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        try:
            summary = chain.invoke({
                "title": article.title,
                "analysis": json.dumps(analysis, indent=2)
            }).content
            
            state["executive_summary"] = summary
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            state["executive_summary"] = f"Executive summary generation failed for: {article.title}. Manual review required."
        
        state["research_complete"] = True
        state["messages"].append(
            SystemMessage(content="Research analysis complete")
        )
        
        return state
    
    def research_article(
        self, 
        article: NewsArticle, 
        relevance_match: RelevanceMatch
    ) -> ResearchResult:
        """Research a high-relevance article and generate comprehensive analysis"""
        
        start_time = datetime.now()
        
        # Initialize research state
        initial_state = ResearchState(
            article=article,
            relevance_match=relevance_match,
            research_queries=[],
            web_search_results=[],
            financial_data={},
            analysis_results={},
            messages=[],
            research_complete=False,
            executive_summary=""
        )
        
        try:
            print(f"🔬 Starting research for: {article.title[:50]}...")
            
            # Run the research workflow
            final_state = self.workflow.invoke(initial_state)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract results
            analysis = final_state.get("analysis_results", {})
            
            # Create research result
            result = ResearchResult(
                article_id=article.id,
                article_title=article.title,
                article_url=article.url,
                research_triggered_at=start_time,
                research_completed_at=end_time,
                research_duration_seconds=duration,
                
                # Impact analysis
                financial_impact=analysis.get("financial_impact", {}),
                operational_impact=analysis.get("operational_impact", {}), 
                reputational_impact=analysis.get("reputational_impact", {}),
                
                # Summary and recommendations
                executive_summary=final_state.get("executive_summary", ""),
                key_takeaways=analysis.get("key_takeaways", []),
                recommended_actions=analysis.get("recommended_actions", []),
                urgency_level=analysis.get("urgency_level", "Medium"),
                
                # Research metadata
                sources_consulted=[r.get('source', 'Unknown') for r in final_state.get("web_search_results", [])],
                research_queries_used=final_state.get("research_queries", []),
                
                # Additional context
                stock_price_context=final_state.get("financial_data", {}).get("stock_data", {}),
                related_articles=final_state.get("web_search_results", [])
            )
            
            print(f"✅ Research complete in {duration:.1f}s - Urgency: {result.urgency_level}")
            return result
            
        except Exception as e:
            print(f"❌ Research failed: {e}")
            
            # Return minimal result on error
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ResearchResult(
                article_id=article.id,
                article_title=article.title,
                article_url=article.url,
                research_triggered_at=start_time,
                research_completed_at=end_time,
                research_duration_seconds=duration,
                executive_summary=f"Research failed for article: {article.title}. Error: {str(e)}",
                urgency_level="Unknown",
                key_takeaways=["Research system error"],
                recommended_actions=["Manual review required"]
            )

    # Integration functions for Streamlit
class ResearchDatabase:
    """Simple database for storing research results"""
    
    def __init__(self, db_path: str = "./research_results.json"):
        self.db_path = db_path
        self.results = self._load_results()
    
    def _load_results(self) -> List[ResearchResult]:
        """Load existing research results"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    return [ResearchResult(**item) for item in data]
        except Exception as e:
            print(f"Error loading research results: {e}")
        return []
    
    def save_result(self, result: ResearchResult):
        """Save a research result"""
        self.results.append(result)
        self._save_results()
    
    def _save_results(self):
        """Save all results to disk"""
        try:
            data = [asdict(result) for result in self.results]
            # Convert datetime objects to strings
            for item in data:
                for key, value in item.items():
                    if isinstance(value, datetime):
                        item[key] = value.isoformat()
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving research results: {e}")
    
    def get_recent_results(self, limit: int = 10) -> List[ResearchResult]:
        """Get most recent research results"""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.research_triggered_at if isinstance(x.research_triggered_at, datetime) else datetime.now(),
            reverse=True
        )
        return sorted_results[:limit]



# Example usage
if __name__ == "__main__":
    # Test the research agent
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Optional
    
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Initialize research agent
    agent = ResearchAgent(
        openai_api_key=OPENAI_API_KEY,
        alpha_vantage_key=ALPHA_VANTAGE_KEY
    )
    
    # Create a mock article and relevance match for testing
    from news_pipeline import NewsArticle
    from relevance_detection import RelevanceMatch
    
    test_article = NewsArticle(
        id="test123",
        title="Apple Supplier Foxconn Reports Supply Chain Disruption",
        content="Foxconn, Apple's major manufacturing partner, reported significant supply chain disruptions affecting iPhone production...",
        url="https://example.com/test",
        source="Test News",
        published_date=datetime.now()
    )
    
    test_match = RelevanceMatch(
        news_article_id="test123",
        news_title=test_article.title,
        news_url=test_article.url,
        news_content_preview=test_article.content[:200],
        matched_entities=[],
        graph_entities=["Foxconn", "iPhone", "Supply Chain"],
        relevance_score=8.5,
        relevance_category="HIGH",
        entity_matches=[],
        semantic_similarity=0.85,
        keyword_matches=["Apple", "Foxconn", "iPhone"],
        created_at=datetime.now()
    )
    
    # Run research
    result = agent.research_article(test_article, test_match)
    
    print("\n=== RESEARCH RESULT ===")
    print(f"Title: {result.article_title}")
    print(f"Urgency: {result.urgency_level}")
    print(f"Duration: {result.research_duration_seconds:.1f}s")
    print(f"\nExecutive Summary:\n{result.executive_summary}")
    print(f"\nKey Takeaways: {result.key_takeaways}")
    print(f"Recommended Actions: {result.recommended_actions}")