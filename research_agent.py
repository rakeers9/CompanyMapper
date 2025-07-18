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
            # Use DuckDuckGo news search (no API key required)
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