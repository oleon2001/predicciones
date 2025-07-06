# ANALIZADOR DE SENTIMIENTOS AVANZADO
"""
Sistema completo de análisis de sentimientos para criptomonedas
Integra noticias, redes sociales, Fear & Greed Index y análisis de texto moderno
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import re
import json
import warnings
warnings.filterwarnings('ignore')

# APIs y web scraping
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tweepy
from newsapi import NewsApiClient
import yfinance as yf
from bs4 import BeautifulSoup

# Análisis de texto
try:
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK no disponible - funcionalidad de sentiment limitada")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers no disponible - usando análisis básico")

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

# Interfaces
from core.interfaces import ISentimentAnalyzer
from config.system_config import SystemConfig

logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer(ISentimentAnalyzer):
    """Analizador de sentimientos avanzado"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.news_api_key = config.api.news_api_key
        self.newsapi = None
        
        # Cache para evitar re-requests
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutos
        
        # Inicializar APIs
        self._initialize_apis()
        
        # Inicializar analizadores de texto
        self._initialize_text_analyzers()
        
        # Keywords para crypto
        self.crypto_keywords = {
            'bitcoin': ['bitcoin', 'btc', 'satoshi'],
            'ethereum': ['ethereum', 'eth', 'ether', 'vitalik'],
            'crypto_general': ['cryptocurrency', 'crypto', 'blockchain', 'defi', 'nft'],
            'positive': ['moon', 'bullish', 'pump', 'rally', 'breakthrough', 'adoption', 'institutional'],
            'negative': ['crash', 'dump', 'bearish', 'regulation', 'ban', 'hack', 'scam']
        }
        
        # Configurar requests con retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _initialize_apis(self):
        """Inicializa APIs externas"""
        if self.news_api_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.news_api_key)
                logger.info("NewsAPI inicializada correctamente")
            except Exception as e:
                logger.warning(f"Error inicializando NewsAPI: {e}")
    
    def _initialize_text_analyzers(self):
        """Inicializa analizadores de texto"""
        self.analyzers = {}
        
        # NLTK VADER
        if NLTK_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                self.analyzers['vader'] = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                
                logger.info("NLTK analizadores inicializados")
            except Exception as e:
                logger.warning(f"Error inicializando NLTK: {e}")
        
        # Transformers (FinBERT para finanzas)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.analyzers['finbert'] = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
                logger.info("FinBERT inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando FinBERT: {e}")
                
        # TextBlob como fallback
        self.analyzers['textblob'] = TextBlob
    
    def analyze_news_sentiment(self, symbol: str, lookback_days: int = 7) -> Dict[str, float]:
        """
        Analiza sentimiento de noticias
        
        Args:
            symbol: Símbolo de criptomoneda
            lookback_days: Días hacia atrás para analizar
            
        Returns:
            Análisis de sentimiento de noticias
        """
        try:
            cache_key = f"news_sentiment_{symbol}_{lookback_days}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Obtener noticias
            articles = self._fetch_news_articles(symbol, lookback_days)
            
            if not articles:
                logger.warning(f"No se encontraron noticias para {symbol}")
                return self._default_sentiment()
            
            logger.info(f"Analizando {len(articles)} artículos para {symbol}")
            
            # Analizar sentimiento de cada artículo
            sentiment_scores = []
            article_analysis = []
            
            for article in articles:
                try:
                    # Combinar título y descripción
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    
                    # Analizar con múltiples métodos
                    scores = self._analyze_text_sentiment(text)
                    sentiment_scores.append(scores)
                    
                    article_analysis.append({
                        'title': article.get('title', ''),
                        'sentiment': scores,
                        'url': article.get('url', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', '')
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analizando artículo: {e}")
                    continue
            
            if not sentiment_scores:
                return self._default_sentiment()
            
            # Agregar resultados
            aggregated_sentiment = self._aggregate_sentiment_scores(sentiment_scores)
            
            # Análisis adicional
            result = {
                **aggregated_sentiment,
                'total_articles': len(articles),
                'analyzed_articles': len(sentiment_scores),
                'time_weighted_sentiment': self._calculate_time_weighted_sentiment(article_analysis),
                'source_diversity': len(set(a.get('source', '') for a in article_analysis)),
                'trend_analysis': self._analyze_sentiment_trend(article_analysis),
                'keyword_analysis': self._analyze_keywords_in_news(articles),
                'articles_sample': article_analysis[:5]  # Muestra de artículos
            }
            
            # Cache
            self._cache_data(cache_key, result)
            
            logger.info(f"Análisis de noticias completado. Sentiment score: {result['composite_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento de noticias: {e}")
            return self._default_sentiment()
    
    def analyze_social_sentiment(self, symbol: str, lookback_days: int = 1) -> Dict[str, float]:
        """
        Analiza sentimiento de redes sociales
        
        Args:
            symbol: Símbolo de criptomoneda
            lookback_days: Días hacia atrás
            
        Returns:
            Análisis de sentimiento social
        """
        try:
            cache_key = f"social_sentiment_{symbol}_{lookback_days}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Simular análisis de redes sociales (en producción usar APIs reales)
            social_data = self._simulate_social_media_data(symbol, lookback_days)
            
            sentiment_analysis = {
                'reddit_sentiment': self._analyze_reddit_sentiment(symbol),
                'twitter_sentiment': self._analyze_twitter_sentiment(symbol),
                'telegram_sentiment': self._analyze_telegram_sentiment(symbol),
                'discord_sentiment': self._analyze_discord_sentiment(symbol),
                'overall_social_score': 0.0,
                'volume_metrics': {
                    'total_mentions': social_data.get('total_mentions', 0),
                    'reddit_posts': social_data.get('reddit_posts', 0),
                    'twitter_tweets': social_data.get('twitter_tweets', 0),
                    'engagement_rate': social_data.get('engagement_rate', 0)
                },
                'trending_topics': social_data.get('trending_topics', []),
                'influencer_sentiment': social_data.get('influencer_sentiment', 0.0)
            }
            
            # Calcular score general
            scores = [
                sentiment_analysis['reddit_sentiment']['score'],
                sentiment_analysis['twitter_sentiment']['score'],
                sentiment_analysis['telegram_sentiment']['score'],
                sentiment_analysis['discord_sentiment']['score']
            ]
            
            sentiment_analysis['overall_social_score'] = np.mean([s for s in scores if s is not None])
            
            # Cache
            self._cache_data(cache_key, sentiment_analysis)
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento social: {e}")
            return self._default_social_sentiment()
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Obtiene Fear & Greed Index real
        
        Returns:
            Datos del Fear & Greed Index
        """
        try:
            cache_key = "fear_greed_index"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # API real del Fear & Greed Index
            try:
                url = "https://api.alternative.me/fng/"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data and len(data['data']) > 0:
                        current = data['data'][0]
                        
                        result = {
                            'value': int(current['value']),
                            'value_classification': current['value_classification'],
                            'timestamp': current['timestamp'],
                            'time_until_update': current.get('time_until_update'),
                            'historical_data': data['data'][:30],  # Últimos 30 días
                            'analysis': self._analyze_fear_greed_trend(data['data'][:30])
                        }
                        
                        # Cache
                        self._cache_data(cache_key, result)
                        
                        logger.info(f"Fear & Greed Index: {result['value']} ({result['value_classification']})")
                        return result
                
            except Exception as e:
                logger.warning(f"Error obteniendo Fear & Greed Index: {e}")
            
            # Fallback: calcular proxy basado en volatilidad de Bitcoin
            btc_data = yf.Ticker("BTC-USD").history(period="30d")
            if not btc_data.empty:
                volatility = btc_data['Close'].pct_change().std() * 100
                
                # Mapear volatilidad a Fear & Greed (aproximación)
                if volatility < 2:
                    fg_value = 75  # Greed
                    classification = "Greed"
                elif volatility < 4:
                    fg_value = 50  # Neutral
                    classification = "Neutral"
                else:
                    fg_value = 25  # Fear
                    classification = "Fear"
                
                result = {
                    'value': fg_value,
                    'value_classification': classification,
                    'timestamp': datetime.now().timestamp(),
                    'method': 'volatility_proxy',
                    'btc_volatility': volatility,
                    'analysis': {'trend': 'neutral', 'confidence': 0.6}
                }
                
                self._cache_data(cache_key, result)
                return result
            
            # Default si todo falla
            return {
                'value': 50,
                'value_classification': 'Neutral',
                'timestamp': datetime.now().timestamp(),
                'method': 'default',
                'analysis': {'trend': 'neutral', 'confidence': 0.3}
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo Fear & Greed Index: {e}")
            return {'value': 50, 'value_classification': 'Neutral', 'error': str(e)}
    
    def _fetch_news_articles(self, symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Obtiene artículos de noticias"""
        articles = []
        
        # Keywords basados en el símbolo
        keywords = self._get_search_keywords(symbol)
        
        if self.newsapi:
            try:
                # Buscar en múltiples queries
                for keyword in keywords[:3]:  # Limitar a 3 para evitar rate limits
                    try:
                        response = self.newsapi.get_everything(
                            q=keyword,
                            language='en',
                            sort_by='relevancy',
                            page_size=20,
                            from_param=(datetime.now() - timedelta(days=lookback_days)).isoformat()
                        )
                        
                        if response['status'] == 'ok':
                            articles.extend(response['articles'])
                        
                    except Exception as e:
                        logger.warning(f"Error buscando {keyword}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"Error con NewsAPI: {e}")
        
        # Eliminar duplicados por URL
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles[:50]  # Limitar a 50 artículos
    
    def _get_search_keywords(self, symbol: str) -> List[str]:
        """Genera keywords de búsqueda"""
        symbol_lower = symbol.lower()
        
        keywords = [symbol]
        
        # Mapear símbolos comunes
        symbol_mapping = {
            'btc': ['bitcoin', 'btc'],
            'eth': ['ethereum', 'eth', 'ether'],
            'ada': ['cardano', 'ada'],
            'sol': ['solana', 'sol'],
            'dot': ['polkadot', 'dot'],
            'link': ['chainlink', 'link'],
            'matic': ['polygon', 'matic'],
            'avax': ['avalanche', 'avax']
        }
        
        if symbol_lower in symbol_mapping:
            keywords.extend(symbol_mapping[symbol_lower])
        
        # Añadir términos generales de crypto
        keywords.append(f"{symbol} cryptocurrency")
        keywords.append(f"{symbol} crypto")
        
        return list(set(keywords))
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analiza sentimiento de texto usando múltiples métodos"""
        scores = {}
        
        # VADER
        if 'vader' in self.analyzers:
            try:
                vader_scores = self.analyzers['vader'].polarity_scores(text)
                scores['vader'] = vader_scores['compound']
            except Exception as e:
                logger.debug(f"Error con VADER: {e}")
        
        # FinBERT
        if 'finbert' in self.analyzers:
            try:
                result = self.analyzers['finbert'](text[:512])  # Limitar longitud
                
                # Convertir a score compuesto
                if result and len(result[0]) > 0:
                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    finbert_score = 0
                    
                    for item in result[0]:
                        label = item['label'].lower()
                        score = item['score']
                        if label in sentiment_map:
                            finbert_score += sentiment_map[label] * score
                    
                    scores['finbert'] = finbert_score
                    
            except Exception as e:
                logger.debug(f"Error con FinBERT: {e}")
        
        # TextBlob
        try:
            blob = TextBlob(text)
            scores['textblob'] = blob.sentiment.polarity
        except Exception as e:
            logger.debug(f"Error con TextBlob: {e}")
        
        # Análisis de keywords específicas de crypto
        scores['keyword'] = self._analyze_crypto_keywords(text)
        
        return scores
    
    def _analyze_crypto_keywords(self, text: str) -> float:
        """Analiza keywords específicas de crypto"""
        text_lower = text.lower()
        
        positive_count = 0
        negative_count = 0
        
        for keyword in self.crypto_keywords['positive']:
            positive_count += text_lower.count(keyword)
        
        for keyword in self.crypto_keywords['negative']:
            negative_count += text_lower.count(keyword)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _aggregate_sentiment_scores(self, sentiment_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Agrega scores de sentimiento"""
        if not sentiment_scores:
            return self._default_sentiment()
        
        # Recopilar todos los scores por método
        methods = {}
        for scores in sentiment_scores:
            for method, score in scores.items():
                if score is not None:
                    if method not in methods:
                        methods[method] = []
                    methods[method].append(score)
        
        # Calcular promedios por método
        method_averages = {}
        for method, scores in methods.items():
            if scores:
                method_averages[method] = np.mean(scores)
        
        # Score compuesto con pesos
        weights = {
            'vader': 0.3,
            'finbert': 0.4,
            'textblob': 0.2,
            'keyword': 0.1
        }
        
        composite_score = 0.0
        total_weight = 0.0
        
        for method, score in method_averages.items():
            weight = weights.get(method, 0.1)
            composite_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            composite_score /= total_weight
        
        return {
            'composite_score': composite_score,
            'method_scores': method_averages,
            'confidence': min(1.0, len(method_averages) / 4.0),  # Confianza basada en métodos disponibles
            'sentiment_classification': self._classify_sentiment(composite_score)
        }
    
    def _classify_sentiment(self, score: float) -> str:
        """Clasifica score numérico en categoría"""
        if score > 0.3:
            return 'Very Positive'
        elif score > 0.1:
            return 'Positive'
        elif score > -0.1:
            return 'Neutral'
        elif score > -0.3:
            return 'Negative'
        else:
            return 'Very Negative'
    
    def _calculate_time_weighted_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Calcula sentimiento ponderado por tiempo"""
        if not articles:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        now = datetime.now()
        
        for article in articles:
            try:
                # Parsear fecha
                published_at = article.get('publishedAt', '')
                if published_at:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    hours_ago = (now - pub_date.replace(tzinfo=None)).total_seconds() / 3600
                    
                    # Peso mayor para noticias más recientes
                    weight = max(0.1, 1.0 / (1.0 + hours_ago / 24.0))  # Decae con tiempo
                else:
                    weight = 0.5  # Peso neutral si no hay fecha
                
                sentiment_score = article.get('sentiment', {}).get('composite_score', 0.0)
                
                total_score += sentiment_score * weight
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"Error calculando peso temporal: {e}")
                continue
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_sentiment_trend(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza tendencia del sentimiento"""
        if len(articles) < 3:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Ordenar por fecha
        sorted_articles = sorted(articles, key=lambda x: x.get('publishedAt', ''))
        
        # Dividir en períodos
        half_point = len(sorted_articles) // 2
        early_articles = sorted_articles[:half_point]
        recent_articles = sorted_articles[half_point:]
        
        # Calcular sentimiento promedio para cada período
        early_sentiment = np.mean([
            a.get('sentiment', {}).get('composite_score', 0.0) 
            for a in early_articles
        ])
        
        recent_sentiment = np.mean([
            a.get('sentiment', {}).get('composite_score', 0.0) 
            for a in recent_articles
        ])
        
        # Determinar tendencia
        sentiment_change = recent_sentiment - early_sentiment
        
        if abs(sentiment_change) < 0.05:
            trend = 'stable'
        elif sentiment_change > 0:
            trend = 'improving'
        else:
            trend = 'deteriorating'
        
        return {
            'trend': trend,
            'sentiment_change': sentiment_change,
            'early_sentiment': early_sentiment,
            'recent_sentiment': recent_sentiment,
            'confidence': min(1.0, len(sorted_articles) / 20.0)
        }
    
    def _analyze_keywords_in_news(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza keywords en noticias"""
        if not articles:
            return {}
        
        # Combinar todo el texto
        all_text = ' '.join([
            f"{article.get('title', '')} {article.get('description', '')}"
            for article in articles
        ]).lower()
        
        # Contar keywords
        keyword_counts = {}
        
        for category, keywords in self.crypto_keywords.items():
            count = sum(all_text.count(keyword) for keyword in keywords)
            keyword_counts[category] = count
        
        # Calcular ratios
        total_positive = keyword_counts.get('positive', 0)
        total_negative = keyword_counts.get('negative', 0)
        
        if total_positive + total_negative > 0:
            positive_ratio = total_positive / (total_positive + total_negative)
        else:
            positive_ratio = 0.5
        
        return {
            'keyword_counts': keyword_counts,
            'positive_ratio': positive_ratio,
            'total_keywords': sum(keyword_counts.values())
        }
    
    def _analyze_fear_greed_trend(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza tendencia del Fear & Greed Index"""
        if len(historical_data) < 7:
            return {'trend': 'insufficient_data'}
        
        values = [int(item['value']) for item in historical_data[:7]]  # Últimos 7 días
        
        # Calcular tendencia
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if abs(slope) < 1:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'  # Hacia greed
        else:
            trend = 'decreasing'  # Hacia fear
        
        # Volatilidad
        volatility = np.std(values)
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'volatility': volatility,
            'current_vs_week_avg': values[0] - np.mean(values),
            'confidence': min(1.0, r_value ** 2)
        }
    
    # Métodos simulados para redes sociales (en producción usar APIs reales)
    
    def _simulate_social_media_data(self, symbol: str, lookback_days: int) -> Dict[str, Any]:
        """Simula datos de redes sociales"""
        # En producción, integrar con APIs reales de Reddit, Twitter, etc.
        return {
            'total_mentions': np.random.randint(100, 1000),
            'reddit_posts': np.random.randint(20, 100),
            'twitter_tweets': np.random.randint(50, 500),
            'engagement_rate': np.random.uniform(0.1, 0.3),
            'trending_topics': ['blockchain', 'defi', 'crypto'],
            'influencer_sentiment': np.random.uniform(-0.5, 0.5)
        }
    
    def _analyze_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analiza sentimiento de Reddit"""
        # Placeholder - en producción usar PRAW
        return {
            'score': np.random.uniform(-0.3, 0.3),
            'posts_analyzed': np.random.randint(10, 50),
            'upvote_ratio': np.random.uniform(0.6, 0.9),
            'confidence': 0.6
        }
    
    def _analyze_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analiza sentimiento de Twitter"""
        # Placeholder - en producción usar tweepy
        return {
            'score': np.random.uniform(-0.2, 0.4),
            'tweets_analyzed': np.random.randint(20, 100),
            'retweet_sentiment': np.random.uniform(-0.1, 0.3),
            'confidence': 0.7
        }
    
    def _analyze_telegram_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analiza sentimiento de Telegram"""
        # Placeholder - en producción usar Telegram API
        return {
            'score': np.random.uniform(-0.1, 0.2),
            'messages_analyzed': np.random.randint(5, 30),
            'group_activity': np.random.uniform(0.3, 0.8),
            'confidence': 0.5
        }
    
    def _analyze_discord_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analiza sentimiento de Discord"""
        # Placeholder - en producción usar Discord API
        return {
            'score': np.random.uniform(-0.2, 0.3),
            'messages_analyzed': np.random.randint(10, 40),
            'server_activity': np.random.uniform(0.2, 0.7),
            'confidence': 0.4
        }
    
    def _default_sentiment(self) -> Dict[str, float]:
        """Sentimiento por defecto"""
        return {
            'composite_score': 0.0,
            'method_scores': {},
            'confidence': 0.0,
            'sentiment_classification': 'Neutral',
            'total_articles': 0,
            'analyzed_articles': 0
        }
    
    def _default_social_sentiment(self) -> Dict[str, Any]:
        """Sentimiento social por defecto"""
        return {
            'reddit_sentiment': {'score': 0.0, 'confidence': 0.0},
            'twitter_sentiment': {'score': 0.0, 'confidence': 0.0},
            'telegram_sentiment': {'score': 0.0, 'confidence': 0.0},
            'discord_sentiment': {'score': 0.0, 'confidence': 0.0},
            'overall_social_score': 0.0,
            'volume_metrics': {'total_mentions': 0}
        }
    
    def _is_cached(self, key: str) -> bool:
        """Verifica cache"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Guarda en cache"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        } 