
import os
import re
import json
import logging
import base64
import threading
from datetime import datetime, timezone
from io import BytesIO
from collections import deque, defaultdict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import faiss
import spacy
import uvicorn
import language_tool_python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from textblob import TextBlob

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, and_
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, SearchParams

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DiabetesAssistant")

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/diabetes_assistant")
#DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/diabetes_assistant")
#DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/diabetes_assistant"
Base = declarative_base()

DATABASE_URL = "sqlite:///./diabetes_assistant.db"


#engine = create_engine(DATABASE_URL, echo=False)
#SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class MedicalRecord(Base):
    __tablename__ = "medical_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    event_type = Column(String, nullable=False)
    details = Column(Text, nullable=False)
    glucose_value = Column(Float, nullable=True)
    medication_data = Column(Text, nullable=True)
    lifestyle_data = Column(Text, nullable=True)
    context_data = Column(Text, nullable=True)
    analysis_results = Column(Text, nullable=True)
    response_data = Column(Text, nullable=True)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    profile_data = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class DiabetesPromptLibrary:
    def __init__(self):
        self.system_context = """You are DiabetesGPT, a medical AI assistant specifically trained in diabetes care with access to real-time patient data..."""
        self.interaction_frames = {
            'glucose_check': {
                'context_requirements': ['recent_readings', 'time_of_day', 'meal_context'],
                'response_template': """Analysis of current reading..."""
            },
            'medication_query': {
                'context_requirements': ['current_medications','last_dose','side_effects'],
                'response_template': """Medication guidance..."""
            },
            'lifestyle_advice': {
                'context_requirements': ['activity_level','dietary_preferences','goals'],
                'response_template': """Lifestyle recommendations..."""
            }
        }

SCENARIO_PROMPTS = {
    "acute_hypoglycemia": "User is experiencing acute low BG symptoms—immediate intake of fast-acting carbohydrates is advised.",
    "nocturnal_hypo": "User reports frequent nocturnal hypoglycemia—suggest adjusting basal insulin or a bedtime snack.",
    "stress_and_bg": "User's stress appears to be affecting their blood glucose. Recommend stress management techniques and relaxation exercises.",
    "exercise_planning": "Provide detailed exercise recommendations tailored for T2D with mild neuropathy.",
    "pump_settings": "User on an insulin pump needs advanced settings for varying activity levels.",
    "type1_new_diagnosis": "User newly diagnosed with type 1 diabetes—explain carb counting, insulin dosing, and daily monitoring.",
    "ramadan_fasting": "Offer guidance for safe fasting during Ramadan, including medication adjustments and monitoring.",
    "pregnancy_gdm": "Advise on postpartum management for a user with gestational diabetes.",
    "retinopathy": "Provide recommendations for screening and managing diabetic retinopathy.",
    "kidney_issues": "User with CKD stage 3 should follow dietary modifications and medication adjustments.",
    "advanced_complications": "User with advanced complications (neuropathy, retinopathy) requires close monitoring and specialized treatment.",
    "injection_sites": "Explain how to rotate injection sites to prevent lipohypertrophy.",
    "multiple_daily_injections": "Advice on balancing basal and bolus insulin for MDI regimens.",
    "CGM_analysis": "Interpret continuous glucose monitoring (CGM) data to identify trends and anomalies.",
    "meal_planning": "Suggest meal composition and timing strategies to maintain stable BG levels.",
    "sick_day_management": "Provide guidelines on managing BG during illness, including hydration and ketone monitoring.",
    "elderly_management": "Simplify diabetes management for elderly patients with comorbidities.",
    "pregnancy_planning": "Discuss preconception BG targets and safe management strategies for diabetes in pregnancy.",
    "disordered_eating": "Address concerns regarding insulin omission for weight control and associated risks.",
    "pre_diabetes": "Recommend lifestyle interventions and possible Metformin use for prediabetes.",
    "long_standing_diabetes": "Provide insights for users with long-standing diabetes regarding complication screening.",
    "foot_care": "Advise on daily foot care routines to prevent ulcers in diabetic neuropathy.",
    "mental_health": "Offer supportive advice for managing the emotional burden of diabetes.",
    "advanced_insulin": "Explain advanced insulin strategies including correction doses and sensitivity factors.",
    "diet_quality": "Discuss balanced diet strategies and nutritional guidelines for diabetes management.",
    "activity_optimization": "Recommend optimal physical activities based on individual BG patterns.",
    "tech_integration": "Explain how to use modern devices (CGM, insulin pumps) for better BG control."
}

class DiabetesPatternAnalyzer:
    def __init__(self):
        self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
    def analyze_glucose_patterns(self, readings, timestamps):
        if not readings or not timestamps:
            return {'current_trend': 0, 'variability': 0, 'predictions': [], 'uncertainty': [], 'risk_patterns': {}}
        try:
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(readings)
            self.gpr.fit(X, y)
            future_times = np.linspace(X[-1], X[-1] + 24, 24)
            preds, std = self.gpr.predict(future_times.reshape(-1, 1), return_std=True)
            return {
                'current_trend': float(np.polyfit(range(len(readings[-10:])), readings[-10:], 1)[0]),
                'variability': float(np.std(readings)),
                'predictions': preds.tolist(),
                'uncertainty': std.tolist(),
                'risk_patterns': self.identify_risk_patterns(readings, timestamps)
            }
        except Exception as e:
            logger.error("Error in analyze_glucose_patterns: " + str(e))
            return {'current_trend': 0, 'variability': 0, 'predictions': [], 'uncertainty': [], 'risk_patterns': {}}
    def detect_postprandial_spikes(self, df, meal_windows=None, spike_threshold=40):
        if meal_windows is None:
            meal_windows = [(7,9), (12,14), (18,20)]
        df = df.sort_values('time').reset_index(drop=True)
        for start_hr, end_hr in meal_windows:
            sub = df[df['time'].dt.hour.between(start_hr, end_hr)]
            if sub.empty:
                continue
            baseline = None
            pre = df[df['time'] < sub.iloc[0]['time']]
            if not pre.empty:
                baseline = pre.iloc[-1]['reading']
            post = df[(df['time'] > sub.iloc[-1]['time']) & (df['time'] <= sub.iloc[-1]['time'] + pd.Timedelta(hours=2))]
            if baseline is not None and not post.empty:
                if (post['reading'].max() - baseline) >= spike_threshold:
                    return True
        return False
    def identify_risk_patterns(self, readings, timestamps):
        df = pd.DataFrame({'reading': readings, 'time': pd.to_datetime(timestamps)})
        morning = df[df.time.dt.hour.between(6, 9)]['reading']
        evening = df[df.time.dt.hour.between(18, 21)]['reading']
        return {
            'dawn_phenomenon': (morning.mean() > evening.mean() + 20 if not morning.empty and not evening.empty else False),
            'nocturnal_lows': (df[df.time.dt.hour.between(0, 5)]['reading'].min() < 70 if not df.empty else False),
            'postprandial_spikes': self.detect_postprandial_spikes(df)
        }
class MedicalContextManager:
    def __init__(self, user_profile_manager, nlp, embed_model='all-MiniLM-L6-v2', embedding_dim=384, context_limit=50):
        self.user_profile_manager = user_profile_manager
        self.nlp = nlp
        self.embedder = SentenceTransformer(embed_model)
        self.embedding_dim = embedding_dim
        self.context_limit = context_limit
        self.user_contexts = defaultdict(lambda: {
            'context_window': deque(maxlen=self.context_limit),
            'index': faiss.IndexFlatL2(self.embedding_dim)
        })
        self.interaction_history = defaultdict(list)
        self.lock = threading.Lock()
    def embed(self, text):
        if isinstance(text, str):
            text = [text]
        embedding = self.embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype('float32')
    def update_context(self, new_context, user_id):
        emb = self.embed(new_context)
        with self.lock:
            user_data = self.user_contexts[user_id]
            try:
                user_data['index'].add(emb.reshape(1, -1))
            except Exception as e:
                logger.error(f"Error adding embedding to FAISS index for user {user_id}: {e}")
            user_data['context_window'].append(new_context)
    def get_relevant_context(self, query, k=5, user_id='default_user'):
        user_data = self.user_contexts[user_id]
        if not user_data['context_window'] or user_data['index'].ntotal == 0:
            return []
        query_emb = self.embed(query)
        with self.lock:
            D, I = user_data['index'].search(query_emb, k)
        relevant_contexts = []
        for idx in I[0]:
            if idx < len(user_data['context_window']):
                relevant_contexts.append(user_data['context_window'][idx])
        return relevant_contexts
    def process_context(self, query, user_data):
        user_id = user_data.get('user_id', 'default_user')
        return {
            'recent_contexts': self.get_recent_interactions(user_id),
            'glucose_patterns': self.analyze_patterns(user_data.get('glucose_history', [])),
            'behavioral_context': self.extract_behavioral_patterns(user_data),
            'medical_context': self.get_medical_context(user_data),
            'interaction_style': self.determine_interaction_style(user_id)
        }
    def get_recent_interactions(self, user_id, limit=5):
        return self.interaction_history[user_id][-limit:]
    def analyze_patterns(self, glucose_history):
        if not glucose_history:
            return {}
        readings = [record['glucose_value'] for record in glucose_history]
        timestamps = [record['timestamp'] for record in glucose_history]
        if len(readings) < 2:
            trends = {'slope': 0.0, 'intercept': readings[0] if readings else 0.0}
        else:
            x = np.arange(len(readings))
            y = np.array(readings)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            trends = {'slope': slope, 'intercept': intercept}
        variability = float(np.std(readings)) if len(readings) > 1 else 0.0
        time_in_range = self.calculate_time_in_range(readings)
        return {
            'trends': trends,
            'variability': variability,
            'time_in_range': time_in_range
        }
    def calculate_time_in_range(self, readings, lower=70, upper=180):
        if not readings:
            return {'in_range_percentage': 0.0}
        in_range = [lower <= r <= upper for r in readings]
        percentage = (sum(in_range) / len(readings)) * 100
        return {'in_range_percentage': round(percentage, 2)}
    def extract_medication_info(self, text):
        medications = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['DRUG', 'PRODUCT']:
                medications.append(ent.text)
        return medications if medications else None
    def extract_lifestyle_info(self, text):
        lifestyle = {}
        doc = self.nlp(text)
        activities = []
        for ent in doc.ents:
            if ent.label_ in ['ACTIVITY', 'ORG']:
                activities.append(ent.text)
        if activities:
            lifestyle['activity_level'] = activities
        diets = []
        for ent in doc.ents:
            if ent.label_ in ['DIET']:
                diets.append(ent.text)
        if diets:
            lifestyle['dietary_preferences'] = diets
        return lifestyle if lifestyle else None
    def extract_behavioral_patterns(self, user_data):
        return {
            'measurement_consistency': self.analyze_measurement_patterns(user_data.get('glucose_history', [])),
            'medication_adherence': self.analyze_medication_adherence(user_data.get('medication_data', '')),
            'lifestyle_patterns': self.analyze_lifestyle_patterns(user_data.get('context_data', ''))
        }
    def analyze_measurement_patterns(self, glucose_history):
        if not glucose_history:
            return {'consistency_score': 0.0}
        timestamps = pd.to_datetime([record['timestamp'] for record in glucose_history])
        intervals = np.diff(timestamps).astype('timedelta64[m]').astype(int)
        if len(intervals) == 0:
            return {'consistency_score': 0.0}
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        consistency = max(0.0, 100 - (std_interval / avg_interval * 100)) if avg_interval > 0 else 0.0
        return {'consistency_score': round(consistency, 2)}
    def analyze_medication_adherence(self, medication_data_json):
        if not medication_data_json:
            return {'adherence_rate': 0.0}
        medication_data = json.loads(medication_data_json)
        taken = sum(1 for med in medication_data if med.get('taken', False))
        total = len(medication_data)
        adherence = (taken / total) * 100 if total > 0 else 0.0
        return {'adherence_rate': round(adherence, 2)}
    def analyze_lifestyle_patterns(self, context_data_json):
        if not context_data_json:
            return {'activity_level': 'unknown', 'diet_quality': 'unknown'}
        context_data = json.loads(context_data_json)
        activity_levels = context_data.get('activity_level', [])
        diet_preferences = context_data.get('dietary_preferences', [])
        activity_counts = defaultdict(int)
        diet_counts = defaultdict(int)
        for activity in activity_levels:
            activity_counts[activity] += 1
        for diet in diet_preferences:
            diet_counts[diet] += 1
        predominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else 'unknown'
        predominant_diet = max(diet_counts, key=diet_counts.get) if diet_counts else 'unknown'
        return {
            'activity_level': predominant_activity,
            'diet_quality': predominant_diet
        }
    def determine_interaction_style(self, user_id):
        interactions = self.interaction_history[user_id]
        if not interactions:
            return 'neutral'
        positive = sum(1 for interaction in interactions if self.is_positive(interaction))
        negative = sum(1 for interaction in interactions if self.is_negative(interaction))
        if positive > negative:
            return 'empathetic'
        elif negative > positive:
            return 'direct'
        else:
            return 'neutral'
    def is_positive(self, text):
        positive_keywords = ['good', 'great', 'thank', 'well', 'happy', 'excellent', 'improved', 'better']
        return any(word in text.lower() for word in positive_keywords)
    def is_negative(self, text):
        negative_keywords = ['bad', 'sad', 'angry', 'frustrated', 'poor', 'hate', 'worse', 'problem']
        return any(word in text.lower() for word in negative_keywords)
    def get_medical_context(self, user_data):
        analysis_results_json = user_data.get('analysis_results', '[]')
        analysis_results = json.loads(analysis_results_json)
        if not analysis_results:
            return {'latest_medical_info': 'None'}
        latest_record = max(analysis_results, key=lambda x: x.get('timestamp', ''))
        return {
            'latest_medical_info': latest_record.get('details', 'None'),
            'last_update': latest_record.get('timestamp', datetime.utcnow().isoformat())
        }
    def add_medical_record(self, user_id, record):
        snippet = f"Medical Record: {record.get('details', '')}"
        self.update_context(snippet, user_id)
    def add_interaction(self, user_id, user_message, assistant_response):
        user_snippet = f"User: {user_message[:300]}"
        assistant_snippet = f"Assistant: {assistant_response}"
        self.update_context(user_snippet, user_id)
        self.update_context(assistant_snippet, user_id)
        self.interaction_history[user_id].append(user_snippet)
        self.interaction_history[user_id].append(assistant_snippet)
    def clear_user_context(self, user_id):
        with self.lock:
            self.user_contexts[user_id]['context_window'].clear()
            self.user_contexts[user_id]['index'].reset()
            self.interaction_history[user_id].clear()
    def save_context_to_db(self, db_session: Session, user_id):
        context_data = {
            'interactions': self.interaction_history[user_id],
            'context_window': list(self.user_contexts[user_id]['context_window'])
        }
        profile_manager = UserProfileManager(db_session)
        profile_manager.update_profile(user_id, {'context_data': context_data})
    def load_context_from_db(self, db_session: Session, user_id):
        profile_manager = UserProfileManager(db_session)
        profile = profile_manager.get_profile(user_id)
        if not profile:
            return
        context_data = profile.get('context_data', {})
        interactions = context_data.get('interactions', [])
        for interaction in interactions:
            self.interaction_history[user_id].append(interaction)
            self.update_context(interaction, user_id)
        window = context_data.get('context_window', [])
        for ctx in window:
            self.update_context(ctx, user_id)
    def search_medical_records(self, db_session: Session, user_id, query, k=5):
        records = db_session.query(MedicalRecord).filter(
            MedicalRecord.user_id == user_id
        ).order_by(MedicalRecord.timestamp.desc()).all()
        if not records:
            logger.debug(f"No medical records found for user_id: {user_id}")
            return []
        snippets = [record.details for record in records]
        if isinstance(snippets, str):
            snippets = [snippets]
        embeddings = self.embedder.encode(snippets, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        with self.lock:
            try:
                self.user_contexts[user_id]['index'].reset()
                self.user_contexts[user_id]['index'].add(embeddings)
            except Exception as e:
                logger.error(f"Error updating FAISS index for user_id {user_id}: {e}")
                return []
        query_emb = self.embed(query)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        logger.debug(f"Query embedding shape: {query_emb.shape}, FAISS index size: {self.user_contexts[user_id]['index'].ntotal}")
        with self.lock:
            try:
                D, I = self.user_contexts[user_id]['index'].search(query_emb, k)
            except Exception as e:
                logger.error(f"Error during FAISS search for user_id {user_id}: {e}")
                relevant_records = []
        relevant_records = []
        for idx in I[0]:
            if idx < len(snippets):
                relevant_records.append(snippets[idx])
        return relevant_records
class DiabetesKnowledgeIntegrator:
    def __init__(self):
        self.treatment_guidelines = self.load_guidelines()
        self.medication_database = self.load_medication_data()
        self.complication_patterns = self.load_complication_patterns()
    def load_guidelines(self):
        return {'type1': {'glucose_targets': {'fasting': '80-130', 'postprandial': '<180'}}, 'type2': {'glucose_targets': {'fasting': '80-130', 'postprandial': '<180'}}}
    def load_medication_data(self):
        return {'Metformin': {'management_notes': ['Start low dose etc'], 'interactions': ['Alcohol caution']}}
    def load_complication_patterns(self):
        return {'hypoglycemia': {'risk_factors': ['Glucose <70'], 'risk_score_weight': 0.4}}

class UserProfileManager:
    def __init__(self, db_session):
        self.db = db_session
        self.cache = {}
    def get_profile(self, user_id):
        if user_id in self.cache:
            return self.cache[user_id]
        profile = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile:
            self.cache[user_id] = json.loads(profile.profile_data)
            return self.cache[user_id]
        
        default_profile = {
            "glucose_readings": [],
            "reading_timestamps": [],
            "medication_data": [],
            "lifestyle_data": [],
            "last_updated": datetime.utcnow().isoformat()
        }
        return default_profile

    def update_profile(self, user_id, interaction_data):
        profile = self.get_profile(user_id)
        if 'glucose_readings' not in profile:
            profile['glucose_readings'] = []
        if 'reading_timestamps' not in profile:
            profile['reading_timestamps'] = []
        if 'medication_data' not in profile:
            profile['medication_data'] = []
        if 'lifestyle_data' not in profile:
            profile['lifestyle_data'] = []
        profile.update(interaction_data)
        if 'medication_data' in profile and isinstance(profile['medication_data'], list):
            taken = sum(1 for med in profile['medication_data'] if med.get('taken', False))
            total = len(profile['medication_data'])
            adherence_rate = (taken / total) * 100 if total > 0 else 0.0
            profile['adherence_rate'] = round(adherence_rate, 2)
        profile['behavioral_insights'] = {
            'activity_level': self.get_predominant_activity_level(profile.get('lifestyle_data', [])),
            'diet_quality': self.get_predominant_diet_quality(profile.get('lifestyle_data', []))
        }
        profile['last_updated'] = datetime.now().isoformat()
        profile['interaction_patterns'] = self.extract_patterns(interaction_data)
        profile['communication_preferences'] = self.infer_preferences(interaction_data)
        profile['medical_history'] = self.update_medical_history(interaction_data)
        self.save_profile(user_id, profile)
        self.cache[user_id] = profile
    def get_predominant_activity_level(self, lifestyle_data):
        activity_levels = [entry.get('activity_level', []) for entry in lifestyle_data]
        flattened = [act for sub in activity_levels for act in sub]
        if not flattened:
            return 'unknown'
        counts = defaultdict(int)
        for act in flattened:
            counts[act] += 1
        return max(counts, key=counts.get)
    def get_predominant_diet_quality(self, lifestyle_data):
        diet_preferences = [entry.get('diet_quality', []) for entry in lifestyle_data]
        flattened = [d for sub in diet_preferences for d in sub]
        if not flattened:
            return 'unknown'
        counts = defaultdict(int)
        for d in flattened:
            counts[d] += 1
        return max(counts, key=counts.get)
    def save_profile(self, user_id, profile_data):
        profile = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            self.db.add(profile)
        profile.profile_data = json.dumps(profile_data)
        self.db.commit()
    def extract_patterns(self, interaction_data):
        interactions = [it['content'] for it in interaction_data.get('interaction_history', [])]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf = vectorizer.fit_transform(interactions)
        nmf = NMF(n_components=5, random_state=42)
        nmf.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()
        patterns = {}
        for idx, topic in enumerate(nmf.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            patterns[f'topic_{idx+1}'] = top_features
        return patterns
    def infer_preferences(self, interaction_data):
        interactions = [it['content'] for it in interaction_data.get('interaction_history', [])]
        sentiments = [TextBlob(t).sentiment.polarity for t in interactions]
        avg = np.mean(sentiments) if sentiments else 0
        if avg > 0.2:
            style = 'empathetic'
        elif avg < -0.2:
            style = 'direct'
        else:
            style = 'neutral'
        return {'communication_preferences': style}
    def update_medical_history(self, interaction_data):
        medical_history = self.user_profile.get('medical_history', [])
        new_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': interaction_data.get('event_type', 'interaction'),
            'details': interaction_data.get('details', ''),
            'glucose_level': interaction_data.get('glucose_value'),
            'medication': interaction_data.get('medication_data', []),
            'lifestyle': interaction_data.get('lifestyle_data', {})
        }
        medical_history.append(new_event)
        self.user_profile['medical_history'] = medical_history[-100:]
        return medical_history
class DiabetesKnowledgeIntegrator:
    def __init__(self):
        self.treatment_guidelines = self.load_guidelines()
        self.medication_database = self.load_medication_data()
        self.complication_patterns = self.load_complication_patterns()
    def load_guidelines(self):
        return {'type1': {'glucose_targets': {'fasting': '80-130', 'postprandial': '<180'}}, 'type2': {'glucose_targets': {'fasting': '80-130', 'postprandial': '<180'}}}
    def load_medication_data(self):
        return {'Metformin': {'management_notes': ['Start low dose etc'], 'interactions': ['Alcohol caution']}}
    def load_complication_patterns(self):
        return {'hypoglycemia': {'risk_factors': ['Glucose <70'], 'risk_score_weight': 0.4}}

class ExtendedIKAD:
    def __init__(self, model_name, user_profile_manager):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.user_profile_manager = user_profile_manager 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.prompt_library = DiabetesPromptLibrary()
        self.pattern_analyzer = DiabetesPatternAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        self.context_manager = MedicalContextManager(self.user_profile_manager, self.nlp)
        self.knowledge_integrator = DiabetesKnowledgeIntegrator()
        self.conversation_state = {'focus': None, 'concerns': set()}
        self.user_profile = {}
        self.lang_tool = language_tool_python.LanguageTool('en-US')
        self.lock = threading.Lock()
        logger.debug("ExtendedIKAD initialized.")
    def load_user_profile(self, db, user_id):
        row = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if row:
            return json.loads(row.profile_data)
        else:
            return {}
    def save_user_profile(self, db, user_id, profile_data):
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.add(profile)
        profile.profile_data = json.dumps(profile_data)
        db.commit()
        logger.debug(f"User profile for {user_id} saved successfully.")
    def process_query(self, query, user_data=None):
        db = SessionLocal()
        logger.debug(f"Current user profile before update: {self.user_profile}")
        user_id = user_data.get('user_id','default_user')
        sess_id = user_data.get('session_id','default_session')
        logger.debug(f"Processing query for user_id: {user_id}, session_id: {sess_id}")
        self.user_profile = self.load_user_profile(db, user_id)
        logger.debug(f"Loaded user profile: {self.user_profile}")
        recs = db.query(MedicalRecord).filter(MedicalRecord.user_id == user_id).order_by(MedicalRecord.timestamp.desc()).limit(10).all()
        for r in reversed(recs):
            snippet = f"User: {r.details[:300]}"
            self.context_manager.update_context(snippet, user_id)
        glucose_value = None
        ent = self.extract_entities(query)
        for entity_text, entity_label in ent:
            match = re.search(r'\b(\d{2,3})\s?(mg/d[lL]|mgdl)\b', entity_text.lower())
            if match:
                glucose_value = float(match.group(1))
                logger.debug(f"Extracted glucose value from entity: {glucose_value} mg/dL")
                break
        if not glucose_value:
            mat = re.search(r'\b(\d{2,3})\s?(mg/d[lL]|mgdl)\b', query.lower())
            if mat:
                glucose_value = float(mat.group(1))
                logger.debug(f"Extracted glucose value from regex: {glucose_value} mg/dL")
        if glucose_value and 40 <= glucose_value <= 600:
            self.user_profile.setdefault('glucose_readings',[]).append(glucose_value)
            self.user_profile.setdefault('reading_timestamps',[]).append(datetime.now(timezone.utc).isoformat())
            logger.debug(f"Stored new glucose value {glucose_value} mg/dL in user profile.")
        else:
            logger.debug("No valid glucose value found or out of realistic range.")
        cxt = self.build_context(query, user_data, user_id)
        cxt['current_glucose'] = glucose_value if glucose_value else None
        cxt['user_query'] = query
        medication_info = self.extract_medication_info(query)
        if medication_info:
            self.user_profile.setdefault('medication_data',[]).extend(medication_info)
            logger.debug(f"Extracted medication info: {medication_info}")
        lifestyle_info = self.extract_lifestyle_info(query)
        if lifestyle_info:
            self.user_profile.setdefault('lifestyle_data',[]).append(lifestyle_info)
            logger.debug(f"Extracted lifestyle info: {lifestyle_info}")
        anl = self.analyze_medical_context(cxt)
        cxt['analysis'] = anl
        prompt = self.construct_prompt(query, cxt, anl)
        raw = self.generate_response(prompt)
        ans = self.post_process(raw, cxt)
        self.context_manager.add_interaction(user_id, query, ans)
        logger.debug(f"Added interaction to context for user_id: {user_id}")
        self.save_user_profile(db, user_id, self.user_profile)
        self._save_record(db, user_id, sess_id, query, ans, cxt, anl)
        logger.debug(f"Saved MedicalRecord for user_id: {user_id}")
        return ans
    def _save_record(self, db, user_id, sess_id, query, resp, cxt, analysis):
        r = MedicalRecord(
            user_id = user_id,
            session_id = sess_id,
            timestamp = datetime.now(timezone.utc),
            event_type = cxt['query_type'],
            details = query,
            glucose_value = cxt.get('current_glucose'),
            medication_data = json.dumps(cxt.get('medication_data'), default=str) if cxt.get('medication_data') else None,
            lifestyle_data = json.dumps(cxt.get('lifestyle_data'), default=str) if cxt.get('lifestyle_data') else None,
            context_data = json.dumps(cxt, default=str),
            analysis_results = json.dumps(analysis, default=str),
            response_data = resp
        )
        db.add(r)
        db.commit()
    def extract_medication_info(self, text):
        medications = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['DRUG', 'PRODUCT']:
                medications.append(ent.text)
        return medications if medications else None
    def extract_lifestyle_info(self, text):
        lifestyle = {}
        doc = self.nlp(text)
        acts = []
        for ent in doc.ents:
            if ent.label_ in ['ACTIVITY', 'ORG']:
                acts.append(ent.text)
        if acts:
            lifestyle['activity_level'] = acts
        diets = []
        for ent in doc.ents:
            if ent.label_ in ['DIET']:
                diets.append(ent.text)
        if diets:
            lifestyle['dietary_preferences'] = diets
        return lifestyle if lifestyle else None
    def extract_entities(self, text):
        d = self.nlp(text)
        return [(e.text, e.label_) for e in d.ents]
    def build_context(self, query, user_data, user_id):
        recs = self.context_manager.get_relevant_context(query, 5, user_id)
        return {
            'query_type': self.classify_query(query),
            'recent_contexts': recs,
            'user_profile': self.user_profile
        }
    def classify_query(self, query):
        pat = {
            'glucose_check': r'\b(glucose|reading|level)\b',
            'medication': r'\b(medication|insulin|pill)\b'
        }
        for k, v in pat.items():
            if re.search(v, query.lower()):
                return k
        return 'general'
    def analyze_medical_context(self, c):
        out = {'risk_level': {}, 'treatment_adherence': {}, 'pattern_insights': None, 'recommended_actions': []}
        if 'glucose_readings' in self.user_profile and 'reading_timestamps' in self.user_profile:
            out['pattern_insights'] = self.pattern_analyzer.analyze_glucose_patterns(
                self.user_profile['glucose_readings'], self.user_profile['reading_timestamps']
            )
        return out
    def construct_prompt(self, query, c, analysis):
        context = self.context_manager.process_context(query, c)
        i_style = context.get('interaction_style', 'neutral')
        med_ctx = context.get('medical_context', {})
        gpats = context.get('glucose_patterns', {})
        rec_ctx = context.get('recent_contexts', [])
        prompt = (
            f"You are a diabetes management AI assistant with real-time data.\n"
            f"Current Context: {json.dumps(med_ctx)}\n"
            f"Interaction Style: {i_style}\n"
            f"User Patterns: {json.dumps(gpats)}\n"
            f"Previous Context: {json.dumps(rec_ctx)}\n"
            f"User Query: {query}\n"
            f"Response Guidelines:\n"
            f"- Maintain consistent persona\n"
            f"- Use {i_style} style\n"
            f"- Provide actionable insights\n"
            f"- Express empathy when appropriate\n"
            f"- Respond in bullet points in Markdown\n"
            f"- Respond in English only\n"
            f"Assistant:"
        )
        return prompt
    def generate_response(self, prompt):
        inp = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inp,
            max_new_tokens=500,
            num_return_sequences=2,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        cands = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        best = self.select_best_response(cands)
        return self.refine(best)
    def select_best_response(self, cands):
        scores = []
        for resp in cands:
            sc = 0
            sc += len(resp.split()) / 100
            sc += resp.count('glucose') + resp.count('diabetes')
            sc += 1 if not any(phrase in resp.lower() for phrase in ['i think','maybe','probably']) else 0
            scores.append(sc)
        return cands[np.argmax(scores)]
    def refine(self, txt):
        r = txt.replace("I think", "Based on data").replace("you should", "it is recommended to")
        return r
    def enforce_bullet_points(self, text):
        lines = text.split("\n")
        formatted = []
        for line in lines:
            if line.strip().startswith("-"):
                formatted.append(line.strip())
            else:
                formatted.append(f"- {line.strip()}.")
        return "\n".join(formatted)
    def post_process(self, resp, c):
        user_query = c.get('user_query', '').lower()
        if re.search(r'\b(frustrated|angry|upset)\b', user_query):
            self.user_profile['preferred_style'] = 'empathetic'
        else:
            self.user_profile['preferred_style'] = 'neutral'
        bullet_resp = self.enforce_bullet_points(resp)
        bullet_resp = self.remove_system_prompt(bullet_resp)
        bullet_resp = self.remove_non_english(bullet_resp)
        try:
            if detect(bullet_resp) != 'en':
                bullet_resp = "I apologize for the confusion. Please provide your query in English."
        except:
            bullet_resp = "I apologize for the confusion. Please provide your query in English."
        c2 = self._remove_rep(bullet_resp)
        c2 = self._gram_check(c2)
        return c2
    def remove_non_english(self, text):
        return ''.join(ch for ch in text if ord(ch) < 128)
    def remove_system_prompt(self, text):
        lines = text.split("\n")
        filtered = []
        for l in lines:
            if l.strip().startswith("System:") or l.strip().startswith("Assistant:"):
                continue
            filtered.append(l)
        return "\n".join(filtered)
    def _remove_rep(self, txt):
        t = txt.split()
        s = set()
        o = []
        i = 0
        while i < len(t):
            e = tuple(t[i:i+3])
            if e in s:
                i += 3
                continue
            s.add(e)
            o.extend(e)
            i += 3
        return " ".join(o)
    def _gram_check(self, txt):
        matches = self.lang_tool.check(txt)
        return language_tool_python.utils.correct(txt, matches)

assistant_singleton = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.post("/api/chat")
def chat_ep(data: ChatRequest, db=Depends(get_db)):
    uid = data.user_id or "default_user"
    sid = data.session_id or "default_session"
    reply = assistant_singleton.process_query(data.query, {'user_id': uid, 'session_id': sid})
    return {"assistant_response": reply}

@app.get("/api/user/profile")
def get_user_profile(user_id: str, db=Depends(get_db)):
    pf = assistant_singleton.user_profile_manager.get_profile(user_id)
    return pf

@app.post("/api/user/profile")
def update_user_profile(user_id: str, profile_data: dict, db=Depends(get_db)):
    assistant_singleton.user_profile_manager.update_profile(user_id, profile_data)
    return {"status": "success"}

@app.get("/api/analysis/patterns")
def get_glucose_patterns(user_id: str, db=Depends(get_db)):
    pf = assistant_singleton.user_profile_manager.get_profile(user_id)
    pat = assistant_singleton.pattern_analyzer.analyze_glucose_patterns(
        pf.get('glucose_readings', []),
        pf.get('reading_timestamps', [])
    )
    return pat

@app.get("/api/visualize")
def visualize_glucose(user_id: str, session_id: str="default_session", fmt: str="json", db=Depends(get_db)):
    pf = assistant_singleton.load_user_profile(db, user_id)
    r = pf.get("glucose_readings", [])
    t = pf.get("reading_timestamps", [])
    if not r or not t:
        raise HTTPException(400, "No glucose data")
    pat = assistant_singleton.pattern_analyzer.analyze_glucose_patterns(r, t)
    if fmt == "json":
        return pat
    elif fmt == "png":
        fig, ax = plt.subplots()
        ax.plot(t, r, 'bo-', label="Glucose")
        p = pat.get("predictions", [])
        if p:
            fx = range(int(t[-1]), int(t[-1]) + 24)
            ax.plot(fx, p, 'r--', label="Predicted")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return FileResponse(buf, media_type="image/png")
    raise HTTPException(400, "Invalid format")

@app.get("/api/medical_records")
def get_medical_records(user_id: str, db: Session = Depends(get_db)):
    recs = db.query(MedicalRecord).filter(MedicalRecord.user_id == user_id).order_by(MedicalRecord.timestamp.asc()).all()
    out = []
    for r in recs:
        out.append({
            'id': r.id,
            'user_id': r.user_id,
            'session_id': r.session_id,
            'timestamp': r.timestamp.isoformat(),
            'event_type': r.event_type,
            'details': r.details,
            'glucose_value': r.glucose_value,
            'medication_data': json.loads(r.medication_data) if r.medication_data else [],
            'lifestyle_data': json.loads(r.lifestyle_data) if r.lifestyle_data else [],
            'response_data': r.response_data
        })
    return out

@app.get("/api/recommendations")
def get_recommendations(user_id: str, db: Session = Depends(get_db)):
    pf = assistant_singleton.user_profile_manager.get_profile(user_id)
    recs = []
    avg_glucose = np.mean(pf.get('glucose_readings', [])) if pf.get('glucose_readings') else None
    if avg_glucose:
        if avg_glucose > 200:
            recs.append("- **Alert:** Your average BG is very high. Please consult your provider immediately.")
        elif avg_glucose > 180:
            recs.append("- Your average BG is high. Consider reviewing your medication and diet.")
        elif avg_glucose < 70:
            recs.append("- Your average BG is low. Monitor for hypoglycemia.")
        else:
            recs.append("- Your average BG is within target range. Good job!")
    adherence_rate = pf.get('adherence_rate', 100.0)
    if adherence_rate < 80:
        recs.append("- **Medication:** Your adherence is below 80%. Consider setting reminders.")
    activity_level = pf.get('behavioral_insights', {}).get('activity_level', 'unknown')
    diet_quality = pf.get('behavioral_insights', {}).get('diet_quality', 'unknown')
    if activity_level != 'unknown':
        recs.append(f"- **Exercise:** Your predominant activity level is {activity_level}. Regular exercise can help manage your BG.")
    if diet_quality != 'unknown':
        recs.append(f"- **Nutrition:** Your diet quality is {diet_quality}. A balanced diet is essential.")
    risk_patterns = pf.get('risk_patterns', {})
    if risk_patterns.get('dawn_phenomenon', False):
        recs.append("- **Dawn Phenomenon:** Detected. Monitor early morning BG and consult your provider.")
    if risk_patterns.get('nocturnal_lows', False):
        recs.append("- **Nocturnal Lows:** Detected. Avoid skipping evening meals.")
    if risk_patterns.get('postprandial_spikes', False):
        recs.append("- **Postprandial Spikes:** Detected. Consider adjusting meal composition or insulin timing.")
    return recs

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def knowledge_graph_get_d3():
    with graph_driver.session() as session:
        results = session.run("""
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name AS src, b.name AS dst, r.rel AS rel
        """)
        data = results.data()
    nodeset = set()
    links = []
    for row in data:
        src = row.get("src")
        dst = row.get("dst")
        rel = row.get("rel")
        if src and dst:
            nodeset.add(src)
            nodeset.add(dst)
            links.append({"source": src, "target": dst, "relation": rel})

    return {"nodes": [{"id": n} for n in nodeset], "links": links}

def knowledge_graph_add_relation(from_node: str, to_node: str, relation: str):
    with graph_driver.session() as session:
        session.write_transaction(
            lambda tx: tx.run(
                "MERGE (a:Entity {name: $from_node}) "
                "MERGE (b:Entity {name: $to_node}) "
                "MERGE (a)-[r:RELATION {rel: $relation}]->(b)",
                from_node=from_node, to_node=to_node, relation=relation
            )
        )


@app.get("/api/knowledge-graph")
def knowledge_graph_ep():
    return knowledge_graph_get_d3()

@app.post("/api/knowledge-graph/add")
def knowledge_graph_add_ep(from_node: str, to_node: str, relation: str):
    knowledge_graph_add_relation(from_node, to_node, relation)
    return {"status": "ok"}

@app.get("/api/scenario/{scenario_name}")
def scenario_ep(scenario_name: str):
    prompt = SCENARIO_PROMPTS.get(scenario_name, "No scenario found for that name.")
    return {"scenario_prompt": prompt}

@app.get("/api/all_users")
def get_all_users(search: Optional[str] = None, sort: Optional[str] = "user_id", order: Optional[str] = "asc", db: Session = Depends(get_db)):
    q = db.query(UserProfile)
    if search:
        q = q.filter(UserProfile.user_id.ilike(f"%{search}%"))
    if sort not in ["user_id", "id"]:
        sort = "user_id"
    q = q.order_by(getattr(UserProfile, sort).desc() if order.lower() == "desc" else getattr(UserProfile, sort))
    profiles = q.all()
    users = [{"user_id": p.user_id, "last_updated": json.loads(p.profile_data).get("last_updated", ""), "profile": json.loads(p.profile_data)} for p in profiles]
    return {"total": len(users), "users": users}


@app.get("/", response_class=HTMLResponse)
def home():
    return '''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Diabetes Companion 2025 - Your Health Guardian</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <meta name="theme-color" media="(prefers-color-scheme: light)" content="#5E46F8" />
    <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#2D3748" />
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <!-- Inline manifest (base64 encoded JSON) -->
    <link rel="manifest" href="data:application/json;base64,ewogICJuYW1lIjogIkRpYWJldGVzIENvbXBhbmlvbiIsCiAgInNob3J0X25hbWUiOiAiRGlhYmV0ZXMiLAogICJzdGFydF91cmwiOiAiLyIsCiAgImRpc3BsYXkiOiAic3RhbmRhbG9uZSIsCiAgImJhY2tncm91bmRfY29sb3IiOiAiI2ZmZmZmZiIsCiAgInRoZW1lX2NvbG9yIjogIiM1RTQ2RjgiCn0=" />

    <!-- Preact Core, Hooks & Router -->
    <script src="https://unpkg.com/preact@10.11.3/dist/preact.umd.js"></script>
    <script src="https://unpkg.com/preact@10.11.3/hooks/dist/hooks.umd.js"></script>
    <script src="https://unpkg.com/preact-router@4.0.1/dist/preact-router.umd.js"></script>

    <!-- TailwindCSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- D3, Lodash, Marked -->
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.21/lodash.min.js"></script>
    <script crossorigin src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    
    <!-- Chart.js and GSAP -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
    <!-- favico -->
    <link rel="icon" type="image/x-icon" href="/static/favicon.ico" />

    <!-- Google Fonts: Plus Jakarta Sans, Space Grotesk, JetBrains Mono -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>

    <style>
      :root {
        --base-font: 'Plus Jakarta Sans', system-ui, sans-serif;
        --heading-font: 'Space Grotesk', sans-serif;
        --code-font: 'JetBrains Mono', monospace;
        --t-fast: 150ms;
        --t-normal: 300ms;
        --t-slow: 500ms;
        --ease: cubic-bezier(0.4, 0, 0.2, 1);
        --s1: 0.25rem;
        --s2: 0.5rem;
        --s3: 0.75rem;
        --s4: 1rem;
        --s5: 1.5rem;
        --s6: 2rem;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --text-primary: #1a1a1a;
        --text-secondary: #64748b;
        --accent: #5E46F8;
        --accent-light: #B4A8FF;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --border: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
      }
      [data-theme="dark"] {
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d3748;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border: #3f3f46;
        --accent: #B4A8FF;
        --accent-light: #5E46F8;
      }
      [data-theme="healthcare"] {
        --accent: #06b6d4;
        --accent-light: #67e8f9;
        --success: #059669;
        --warning: #d97706;
        --error: #dc2626;
      }
      body {
        font-family: var(--base-font);
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: background-color var(--t-normal) var(--ease),
                    color var(--t-normal) var(--ease);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        overflow-x: hidden;
      }
      h1, h2, h3, h4, h5, h6 {
        font-family: var(--heading-font);
        font-weight: 600;
        line-height: 1.2;
      }
      code, pre {
        font-family: var(--code-font);
      }
      #root {
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
      }
      .light-theme {
        background: linear-gradient(135deg, #fefefe 0%, #f9fafb 100%);
        color: var(--text-primary);
      }
      .solarized-theme {
        background: #fdf6e3;
        color: var(--text-secondary);
      }
      .material-theme {
        background: #f5f5f5;
        color: #212121;
        background-image: linear-gradient(45deg, rgba(0,0,0,0.02) 25%, transparent 25%),
          linear-gradient(-45deg, rgba(0,0,0,0.02) 25%, transparent 25%);
        background-size: 20px 20px;
      }
      .monokai-theme {
        background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
        color: #333;
      }
      
      header {
        padding: var(--s4) var(--s5);
        background: inherit;
        box-shadow: var(--shadow-md);
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      nav {
        padding: var(--s2) var(--s4);
        background: rgba(0, 0, 0, 0.03);
      }
      main {
        flex: 1;
        overflow-y: auto;
        padding: var(--s4);
      }
      .app-container {
        max-width: 1200px;
        margin: auto;
        padding: var(--s4);
      }
      .card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--s3);
        padding: var(--s4);
        box-shadow: var(--shadow-md);
        margin-bottom: var(--s4);
        transition: transform var(--t-normal) var(--ease);
      }
      .card:hover {
        transform: translateY(-var(--s1));
        box-shadow: var(--shadow-lg);
      }
      .custom-tabs a {
        display: flex;
        align-items: center;
        gap: var(--s2);
        padding: var(--s3) var(--s4);
        border-radius: var(--s3);
        transition: background var(--t-normal) var(--ease), transform var(--t-normal) var(--ease);
        font-weight: 500;
      }
      .custom-tabs a:hover {
        background: rgba(0, 0, 0, 0.08);
        transform: scale(1.03);
      }
      .chat-bubble {
        padding: var(--s4);
        border-radius: var(--s3);
        box-shadow: var(--shadow-md);
        max-width: 70%;
        margin-bottom: var(--s4);
        word-wrap: break-word;
        animation: fadeInUp var(--t-normal) var(--ease);
      }
      .chat-bubble.user {
        background: #d1e7dd;
        color: #0f5132;
        align-self: flex-end;
      }
      .chat-bubble.assistant {
        background: var(--bg-primary);
        color: #000;
        align-self: flex-start;
        border: 1px solid var(--border);
      }
      .monokai-theme .chat-bubble.assistant {
        background: #f5f5f5;
        color: #333;
      }
      @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .markdown-content {
        line-height: 1.6;
        white-space: pre-wrap;
      }
      .markdown-content h1,
      .markdown-content h2,
      .markdown-content h3,
      .markdown-content h4 {
        margin: 1em 0 0.5em;
        font-weight: 700;
      }
      .markdown-content p {
        margin: 0.5em 0;
      }
      .markdown-content ul {
        margin: 0.5em 0;
        padding-left: 1.5em;
      }
      .markdown-content li {
        margin-bottom: 0.25em;
      }
      .markdown-content pre {
        background: #2d2d2d;
        color: #f8f8f2;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow: auto;
      }
      .monokai-theme .markdown-content pre {
        background: #272822;
      }
      .markdown-content code {
        background: #e5e7eb;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
      }
      .monokai-theme .markdown-content code {
        background: #49483e;
        color: #f8f8f2;
      }
      .terminate-btn {
        position: absolute;
        top: var(--s2);
        right: var(--s2);
        background: rgba(255, 0, 0, 0.85);
        color: #fff;
        border: none;
        border-radius: var(--s1);
        padding: var(--s1) var(--s2);
        cursor: pointer;
        z-index: 100;
        transition: background var(--t-normal) var(--ease);
      }
      .terminate-btn:hover {
        background: rgba(255, 0, 0, 1);
      }
      .chat-input-container {
        display: flex;
        align-items: center;
        border-top: 1px solid var(--border);
        padding: var(--s3) var(--s4);
        background: var(--bg-primary);
        box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
        border-radius: 0 0 var(--s3) var(--s3);
        transition: background var(--t-normal) var(--ease);
      }
      .chat-input-container input {
        flex: 1;
        padding: var(--s3);
        border: none;
        outline: none;
        font-size: 1rem;
        border-radius: var(--s2) 0 0 var(--s2);
      }
      .chat-input-container button {
        background: #1976d2;
        color: #fff;
        border: none;
        padding: var(--s3) var(--s4);
        cursor: pointer;
        transition: background var(--t-normal) var(--ease);
        border-radius: 0 var(--s2) var(--s2) 0;
      }
      .chat-input-container button:hover {
        background: #1565c0;
      }
      .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 200;
      }
      .modal {
        background: var(--bg-primary);
        padding: var(--s5);
        border-radius: var(--s3);
        box-shadow: var(--shadow-lg);
        opacity: 0;
        transform: scale(0.9);
      }
      .glass-effect {
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.8);
      }
      .vital-card {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: var(--s3);
        align-items: center;
        padding: var(--s3);
        background: var(--bg-secondary);
        border-radius: var(--s2);
        border: 1px solid var(--border);
      }
      .vital-icon {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-light);
        border-radius: 50%;
        color: var(--accent);
      }
      .glucose-graph {
        width: 100%;
        height: 300px;
        background: var(--bg-secondary);
        border-radius: var(--s3);
        padding: var(--s4);
        position: relative;
      }
      .glucose-marker {
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--accent);
        transform: translate(-50%, -50%);
        transition: transform var(--t-normal) var(--ease);
      }
      .glucose-marker:hover {
        transform: translate(-50%, -50%) scale(1.5);
      }
      
      .chat-container {
        scrollbar-width: thin;
        scrollbar-color: var(--accent-light) transparent;
      }
      .chat-container::-webkit-scrollbar {
        width: 6px;
      }
      .chat-container::-webkit-scrollbar-track {
        background: transparent;
      }
      .chat-container::-webkit-scrollbar-thumb {
        background-color: var(--accent-light);
        border-radius: 3px;
      }
      
      .input-wrapper {
        position: relative;
        isolation: isolate;
      }
      .input-wrapper::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(to right, transparent, white 10%);
        z-index: -1;
        opacity: 0.8;
      }
      
      .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,0.7);
        transform: scale(0);
        animation: ripple 600ms linear;
        pointer-events: none;
      }
      @keyframes ripple {
        to {
          transform: scale(4);
          opacity: 0;
        }
      }
      
      @media (max-width: 768px) {
        :root {
          --s4: 0.875rem;
          --s5: 1.25rem;
          --s6: 1.5rem;
        }
        .vital-card {
          grid-template-columns: 1fr;
          text-align: center;
        }
        .vital-icon {
          margin: 0 auto;
        }
      }
      @media (prefers-reduced-motion: reduce) {
        * {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
          scroll-behavior: auto !important;
        }
      }
      
      .ikad-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent);
      }
      .ikad-insights {
        background: var(--bg-secondary);
        border-radius: var(--s3);
        padding: var(--s4);
        box-shadow: var(--shadow);
        animation: fadeInUp var(--t-normal) var(--ease);
      }
      .insights-list {
        list-style: none;
        padding: 0;
      }
      .insights-list li::before {
        content: "• ";
        color: var(--accent);
        margin-right: var(--s1);
      }
    </style>
  </head>
  <body data-theme="ikad" class="light-theme">
    <div id="root"></div>
    
    <div id="modal-backdrop" class="modal-backdrop" style="display:none;">
      <div id="modal" class="modal">
        <h2 class="text-2xl font-bold mb-4">Notification</h2>
        <p class="mb-4">This is the Intelligent Knowledge Assistant for Diabetes.</p>
        <button id="modal-close" class="px-4 py-2 bg-blue-600 text-white rounded">Close</button>
      </div>
    </div>
    
    <script type="text/javascript">
      if ('serviceWorker' in navigator) {
        const swCode = `
          const CACHE_NAME = 'diabetes-dashboard-v1';
          const urlsToCache = ['/', '/index.html'];
          self.addEventListener('install', event => {
            event.waitUntil(
              caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
            );
          });
          self.addEventListener('fetch', event => {
            event.respondWith(
              caches.match(event.request).then(response => response || fetch(event.request))
            );
          });
          self.addEventListener('activate', event => {
            const cacheWhitelist = [CACHE_NAME];
            event.waitUntil(
              caches.keys().then(cacheNames => Promise.all(
                cacheNames.map(cacheName => {
                  if (cacheWhitelist.indexOf(cacheName) === -1) {
                    return caches.delete(cacheName);
                  }
                })
              ))
            );
          });
        `;
        const blob = new Blob([swCode], { type: 'application/javascript' });
        const swUrl = URL.createObjectURL(blob);
        navigator.serviceWorker.register(swUrl).then(registration => {
          console.log('Service Worker registered with scope:', registration.scope);
        }).catch(err => {
          console.error('Service Worker registration failed:', err);
        });
      }
      
      function showModal() {
        const backdrop = document.getElementById("modal-backdrop");
        const modal = document.getElementById("modal");
        backdrop.style.display = "flex";
        modal.animate(
          [
            { opacity: 0, transform: "scale(0.9)" },
            { opacity: 1, transform: "scale(1)" }
          ],
          { duration: 300, fill: "forwards" }
        );
      }
      function hideModal() {
        const backdrop = document.getElementById("modal-backdrop");
        const modal = document.getElementById("modal");
        modal.animate(
          [
            { opacity: 1, transform: "scale(1)" },
            { opacity: 0, transform: "scale(0.9)" }
          ],
          { duration: 300, fill: "forwards" }
        ).onfinish = () => {
          backdrop.style.display = "none";
        };
      }
      document.getElementById("modal-close").addEventListener("click", hideModal);
      
      marked.setOptions({
        breaks: true,
        gfm: true,
        smartLists: true,
        smartypants: true
      });
      
      function trimAssistantResponse(text) {
        const marker = "Assistant:";
        const idx = text.indexOf(marker);
        if (idx >= 0) {
          const response = text.slice(idx + marker.length).trim();
          const latestResponseIdx = response.lastIndexOf(`\n\n`);
          return latestResponseIdx >= 0 ? response.slice(latestResponseIdx + 2) : response;
        }
        return text;
      }
      
      function formatAssistantReply(text) {
        let processed = trimAssistantResponse(text);
        if (!processed.trim().startsWith("-")) {
          processed = processed.split(`\n`)
            .filter(line => line.trim() !== "")
            .map(line => "- " + line.trim())
            .join(`\n`);
        }
        return processed;
      }
      
      const e = preact.h;
      const { render } = preact;
      const { useState, useEffect, useRef, useCallback } = preactHooks;
      const { Router, Link, route } = preactRouter;
      console.log("SOTA Diabetes Management Dashboard Loaded");
      
      function useNaturalTyping(fullText, baseSpeed = 30) {
        const [displayText, setDisplayText] = useState("");
        const indexRef = useRef(0);
        const timeoutRef = useRef(null);
        const typeNext = useCallback(() => {
          if (indexRef.current < fullText.length) {
            setDisplayText(prev => prev + fullText[indexRef.current]);
            indexRef.current++;
            const delay = baseSpeed + Math.floor(Math.random() * 50);
            timeoutRef.current = setTimeout(typeNext, delay);
          }
        }, [fullText, baseSpeed]);
        useEffect(() => {
          indexRef.current = 0;
          setDisplayText("");
          typeNext();
          return () => clearTimeout(timeoutRef.current);
        }, [fullText, typeNext]);
        return { displayText, cancel: () => clearTimeout(timeoutRef.current) };
      }
      
      function TypingMessage({ fullText, baseSpeed = 30 }) {
        const { displayText, cancel } = useNaturalTyping(fullText, baseSpeed);
        const [isTyping, setIsTyping] = useState(true);
        useEffect(() => {
          if (displayText.length === fullText.length) {
            setIsTyping(false);
          }
        }, [displayText, fullText]);
        const stopTyping = () => {
          cancel();
          setIsTyping(false);
        };
        const finalText = isTyping ? displayText : formatAssistantReply(displayText);
        return e(
          "div",
          { className: "relative" },
          isTyping && e("button", { className: "terminate-btn", onClick: stopTyping }, "Stop"),
          e(
            "div",
            { className: "chat-bubble assistant pt-8 transition-all duration-300" },
            e("div", { className: "markdown-content" }, finalText)
          )
        );
      }
      
      function Markdown({ content }) {
        const html = marked.parse(content || "");
        return e("div", {
          className: "markdown-content",
          dangerouslySetInnerHTML: { __html: html }
        });
      }
      
      function ThemeSwitcher({ currentTheme, onThemeChange }) {
        const themes = [
          { name: "Light", class: "light-theme" },
          { name: "Solarized", class: "solarized-theme" },
          { name: "Material", class: "material-theme" },
          { name: "Monokai", class: "monokai-theme" }
        ];
        return e(
          "select",
          {
            value: currentTheme,
            onChange: (ev) => onThemeChange(ev.target.value),
            className: "px-4 py-2 border rounded transition-all duration-300"
          },
          themes.map(t => e("option", { value: t.class, key: t.class }, t.name))
        );
      }
      
      function CustomTabs({ selectedUser }) {
        const tabs = [
          { href: "/all_users", label: "Users", icon: "fas fa-users" },
          { href: `/chat/${selectedUser}`, label: "Chat", icon: "fas fa-comments" },
          { href: `/profile/${selectedUser}`, label: "Profile", icon: "fas fa-user" },
          { href: `/charts/${selectedUser}`, label: "Charts", icon: "fas fa-chart-line" },
          { href: `/recs/${selectedUser}`, label: "Recs", icon: "fas fa-lightbulb" },
          { href: "/kg", label: "KG", icon: "fas fa-network-wired" },
          { href: "/health", label: "Health", icon: "fas fa-heartbeat" },
          { href: "/ikad", label: "IKAD", icon: "fas fa-brain" }
        ];
        return e(
          "nav",
          { className: "custom-tabs flex space-x-4 bg-gray-100 p-2 rounded mb-2" },
          tabs.map(tab =>
            e(
              Link,
              {
                href: tab.href,
                className: "flex items-center gap-2 p-2 hover:bg-gray-200 rounded transition-all duration-300"
              },
              e("i", { className: tab.icon }),
              e("span", null, tab.label)
            )
          )
        );
      }
      
      function AllUsers({ onSelect }) {
        const [users, setUsers] = useState([]);
        const [total, setTotal] = useState(0);
        const [loading, setLoading] = useState(false);
        const [search, setSearch] = useState("");
        const fetchUsers = useCallback(() => {
          setLoading(true);
          setTimeout(() => {
            const sampleUsers = [
              { user_id: "user1", last_updated: "2024-02-01" },
              { user_id: "user2", last_updated: "2024-02-01" }
            ];
            setUsers(sampleUsers);
            setTotal(sampleUsers.length);
            setLoading(false);
          }, 500);
        }, []);
        const debouncedFetch = useCallback(_.debounce(fetchUsers, 500), [fetchUsers]);
        useEffect(() => {
          debouncedFetch();
          return () => debouncedFetch.cancel();
        }, [search, debouncedFetch]);
        const handleUserClick = useCallback(
          (userId) => {
            if (onSelect) onSelect(userId);
            route(`/chat/${userId}`);
          },
          [onSelect]
        );
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "All Users (" + total + ")"),
          e(
            "div",
            { className: "flex space-x-2 mb-4" },
            e("input", {
              type: "text",
              className: "border p-2 rounded",
              placeholder: "Search...",
              value: search,
              onChange: (ev) => setSearch(ev.target.value)
            }),
            e(
              "button",
              { className: "px-4 py-2 bg-blue-600 text-white rounded", onClick: fetchUsers, onClickCapture: createRipple },
              "Refresh"
            )
          ),
          loading
            ? e("p", null, "Loading...")
            : e(
                "ul",
                { className: "space-y-2" },
                users.map(u =>
                  e(
                    "li",
                    {
                      key: u.user_id,
                      className: "bg-white p-3 rounded shadow cursor-pointer hover:bg-blue-100 transition-all duration-300",
                      onClick: () => handleUserClick(u.user_id)
                    },
                    e("div", { className: "font-semibold" }, u.user_id),
                    e("div", { className: "text-sm" }, "Last Updated: " + u.last_updated)
                  )
                )
              )
        );
      }
      
      function ChatTab({ userId }) {
        const [chatHistory, setChatHistory] = useState([]);
        const [chatInput, setChatInput] = useState("");
        const [loading, setLoading] = useState(false);
        useEffect(() => {
          if (userId) {
            const cached = localStorage.getItem("chatHistory_" + userId);
            if (cached) setChatHistory(JSON.parse(cached));
          }
        }, [userId]);
        useEffect(() => {
          if (userId) {
            localStorage.setItem("chatHistory_" + userId, JSON.stringify(chatHistory));
          }
        }, [chatHistory, userId]);
        useEffect(() => {
          if (!userId) return;
          if (!localStorage.getItem("chatHistory_" + userId)) {
            fetch("/api/medical_records?user_id=" + userId)
              .then(res => res.json())
              .then(data => setChatHistory(data))
              .catch(err => console.error("Error fetching chat history:", err));
          }
        }, [userId]);
        function sendMessage() {
          if (!chatInput.trim() || !userId) return;
          setLoading(true);
          const localMsg = { role: "user", content: chatInput };
          setChatHistory(prev => [...prev, localMsg]);
          const msgToSend = chatInput;
          setChatInput("");
          fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: msgToSend, user_id: userId, session_id: "default_session" })
          })
            .then(res => res.json())
            .then(data => {
              const formatted = formatAssistantReply(data.assistant_response);
              setChatHistory(prev => [
                ...prev,
                { role: "assistant", fullText: formatted }
              ]);
              setLoading(false);
            })
            .catch(err => {
              console.error("Error sending message:", err);
              setLoading(false);
            });
        }
        if (!userId) return e("div", { className: "p-4" }, "No user selected.");
        return e(
          "div",
          { className: "p-4 flex flex-col h-full" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Chat with " + userId),
          e(
            "div",
            { className: "flex-1 overflow-auto border p-4 flex flex-col transition-all duration-300 chat-container" },
            chatHistory.map((msg, i) =>
              msg.role === "assistant"
                ? msg.fullText
                  ? e(TypingMessage, { key: i, fullText: msg.fullText })
                  : e(
                      "div",
                      { key: i, className: "chat-bubble assistant transition-all duration-300" },
                      e(Markdown, { content: trimAssistantResponse(msg.content) })
                    )
                : e(
                    "div",
                    { key: i, className: "chat-bubble user transition-all duration-300" },
                    e("div", { className: "markdown-content" }, msg.content)
                  )
            )
          ),
          e(
            "div",
            { className: "chat-input-container" },
            e("div", { className: "input-wrapper w-full" },
              e("input", {
                type: "text",
                placeholder: "Type your message...",
                value: chatInput,
                onChange: (ev) => setChatInput(ev.target.value),
                onKeyDown: (ev) => { if (ev.key === "Enter") sendMessage(); },
                className: "w-full floating-input"
              })
            ),
            e("button", { onClick: sendMessage, disabled: loading, onClickCapture: createRipple }, loading ? "Sending..." : "Send")
          )
        );
      }
      
      function ProfileTab({ userId }) {
        const [profile, setProfile] = useState(null);
        useEffect(() => {
          if (!userId) return;
          fetch("/api/user/profile?user_id=" + userId)
            .then(res => res.json())
            .then(data => setProfile(data))
            .catch(err => console.error("Error fetching profile:", err));
        }, [userId]);
        if (!userId) return e("div", { className: "p-4" }, "No user selected.");
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Profile of " + userId),
          profile
            ? e("pre", { className: "card" }, JSON.stringify(profile, null, 2))
            : e("p", {}, "Loading profile...")
        );
      }
      
      function ChartsTab({ userId }) {
        const [data, setData] = useState(null);
        const chartRef = useRef(null);
        useEffect(() => {
          if (!userId) return;
          fetch("/api/visualize?user_id=" + userId + "&fmt=json")
            .then(res => res.json())
            .then(d => setData(d))
            .catch(err => console.error("Error fetching chart data:", err));
        }, [userId]);
        useEffect(() => {
          if (!data || !data.predictions) return;
          const svg = d3.select(chartRef.current);
          svg.selectAll("*").remove();
          const width = 600, height = 300;
          const margin = { top: 20, right: 20, bottom: 30, left: 50 };
          const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
          const preds = data.predictions || [];
          const x = d3.scaleLinear().domain([0, preds.length - 1]).range([0, width - margin.left - margin.right]);
          const y = d3.scaleLinear().domain([0, d3.max(preds)]).range([height - margin.top - margin.bottom, 0]);
          const line = d3.line().x((_, i) => x(i)).y((d) => y(d));
          g.append("path").datum(preds).attr("fill", "none").attr("stroke", "steelblue").attr("stroke-width", 2).attr("d", line);
          g.append("g").attr("transform", `translate(0,${height - margin.top - margin.bottom})`).call(d3.axisBottom(x));
          g.append("g").call(d3.axisLeft(y));
        }, [data]);
        if (!userId) return e("div", { className: "p-4" }, "No user selected.");
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Charts for " + userId),
          e("svg", { ref: chartRef, width: 600, height: 300, className: "card" })
        );
      }
      
      function RecsTab({ userId }) {
        const [recs, setRecs] = useState([]);
        useEffect(() => {
          if (!userId) return;
          fetch("/api/recommendations?user_id=" + userId)
            .then(res => res.json())
            .then(data => setRecs(data))
            .catch(err => console.error("Error fetching recommendations:", err));
        }, [userId]);
        if (!userId) return e("div", { className: "p-4" }, "No user selected.");
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Recommendations for " + userId),
          recs.length === 0
            ? e("p", {}, "No recommendations available.")
            : e("ul", { className: "list-disc pl-5 space-y-2" }, recs.map((rec, i) => e("li", { key: i }, e(Markdown, { content: rec }))))
        );
      }
      
      function KGTab() {
        const [kgData, setKgData] = useState({ nodes: [], links: [] });
        const svgRef = useRef(null);
        useEffect(() => {
          fetch("/api/knowledge-graph")
            .then(res => res.json())
            .then(data => setKgData(data))
            .catch(err => console.error("Error fetching KG data:", err));
        }, []);
        useEffect(() => {
          if (!kgData || !kgData.nodes || !kgData.nodes.length) return;
          const svg = d3.select(svgRef.current);
          svg.selectAll("*").remove();
          const width = 600, height = 400;
          const simulation = d3.forceSimulation(kgData.nodes)
            .force("charge", d3.forceManyBody().strength(-150))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("link", d3.forceLink(kgData.links).distance(100).id(d => d.id))
            .on("tick", ticked);
          const svgEl = svg.append("svg").attr("width", width).attr("height", height);
          const link = svgEl.selectAll("line").data(kgData.links).enter().append("line")
            .attr("stroke", "#999").attr("stroke-width", 1.5);
          const node = svgEl.selectAll("circle").data(kgData.nodes).enter().append("circle")
            .attr("r", 8).attr("fill", "steelblue")
            .call(d3.drag()
              .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
              .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
              .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
            );
          const label = svgEl.selectAll("text").data(kgData.nodes).enter().append("text")
            .attr("font-size", "10px").attr("dx", 12).attr("dy", ".35em").text(d => d.id);
          function ticked() {
            link.attr("x1", d => (d.source && typeof d.source.x === "number") ? d.source.x : 0)
                .attr("y1", d => (d.source && typeof d.source.y === "number") ? d.source.y : 0)
                .attr("x2", d => (d.target && typeof d.target.x === "number") ? d.target.x : 0)
                .attr("y2", d => (d.target && typeof d.target.y === "number") ? d.target.y : 0);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            label.attr("x", d => d.x).attr("y", d => d.y);
          }
        }, [kgData]);
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Knowledge Graph"),
          e("div", { ref: svgRef, style: { width: "600px", height: "400px", background: "#fff" } })
        );
      }
      
      function GlucoseMonitor({ data }) {
        const [chartInstance, setChartInstance] = useState(null);
        const chartRef = useRef(null);
        useEffect(() => {
          if (!chartRef.current) return;
          const ctx = chartRef.current.getContext('2d');
          const chart = new Chart(ctx, {
            type: 'line',
            data: {
              datasets: [{
                label: 'Glucose Level',
                data: data,
                borderColor: getComputedStyle(document.body).getPropertyValue('--accent'),
                tension: 0.4
              }]
            },
            options: {
              responsive: true,
              interaction: { intersect: false, mode: 'index' },
              plugins: {
                legend: { display: false },
                tooltip: {
                  backgroundColor: getComputedStyle(document.body).getPropertyValue('--bg-secondary'),
                  titleColor: getComputedStyle(document.body).getPropertyValue('--text-primary'),
                  bodyColor: getComputedStyle(document.body).getPropertyValue('--text-secondary'),
                  borderColor: getComputedStyle(document.body).getPropertyValue('--border'),
                  borderWidth: 1
                }
              },
              scales: {
                y: { beginAtZero: true, grid: { color: getComputedStyle(document.body).getPropertyValue('--border') } },
                x: { grid: { display: false } }
              }
            }
          });
          setChartInstance(chart);
          return () => chart.destroy();
        }, [data]);
        return e('canvas', { ref: chartRef, className: 'glucose-graph' });
      }
      
      function VitalCard({ icon, title, value, unit, trend }) {
        return e(
          'div',
          { className: 'vital-card' },
          e('div', { className: 'vital-icon' }, e('i', { className: icon })),
          e(
            'div',
            null,
            e('h3', { className: 'text-lg font-semibold' }, title),
            e(
              'div',
              { className: 'flex items-center gap-2' },
              e('span', { className: 'text-2xl font-bold' }, value),
              e('span', { className: 'text-sm text-secondary' }, unit),
              trend !== undefined &&
                e('i', { className: `fas fa-arrow-${trend > 0 ? 'up text-success' : (trend < 0 ? 'down text-error' : '')}` })
            )
          )
        );
      }
      
      function HealthDashboard({ userId }) {
        const [vitals, setVitals] = useState({
          glucose: 120,
          bloodPressure: "120/80",
          heartRate: 72
        });
        const [glucoseData, setGlucoseData] = useState([110, 115, 120, 125, 130, 128, 122]);
        return e(
          "div",
          { className: "p-4" },
          e("h2", { className: "text-xl font-bold mb-4" }, "Health Dashboard"),
          e(
            "div",
            { className: "grid grid-cols-1 md:grid-cols-3 gap-4" },
            e(VitalCard, { icon: "fas fa-tint", title: "Glucose", value: vitals.glucose, unit: "mg/dL", trend: 1 }),
            e(VitalCard, { icon: "fas fa-heart", title: "Heart Rate", value: vitals.heartRate, unit: "bpm", trend: -1 }),
            e(VitalCard, { icon: "fas fa-tachometer-alt", title: "Blood Pressure", value: vitals.bloodPressure, unit: "", trend: 0 })
          ),
          e("h3", { className: "text-lg font-semibold mt-6 mb-2" }, "Glucose Trend"),
          e(GlucoseMonitor, { data: glucoseData })
        );
      }
      
      function IKADAssistant() {
        const [response, setResponse] = useState(`Assistant: Here are some key insights for managing your diabetes:\nStay hydrated.\nExercise regularly.\nMonitor your blood sugar levels.`);
        const getNewResponse = () => {
          const newReply = `Assistant: New insights:\n1. Maintain a balanced diet.\n2. Get regular check-ups.\n3. Manage stress effectively.`;
          setResponse(newReply);
        };
        return e(
          "div",
          { className: "p-4 ikad-insights" },
          e("h2", { className: "ikad-header mb-4" }, "IKAD Assistant Insights"),
          e(Markdown, { content: formatAssistantReply(response) }),
          e("button", { className: "btn btn-primary mt-4", onClick: getNewResponse, onClickCapture: createRipple }, "Get New Insights")
        );
      }
      
      function Modal({ title, message, onClose }) {
        const modalRef = useRef(null);
        useEffect(() => {
          if (modalRef.current) {
            modalRef.current.animate(
              [
                { opacity: 0, transform: "scale(0.9)" },
                { opacity: 1, transform: "scale(1)" }
              ],
              { duration: 300, fill: "forwards" }
            );
          }
        }, []);
        return e(
          "div",
          {
            className: "modal-backdrop",
            onClick: onClose
          },
          e(
            "div",
            {
              ref: modalRef,
              className: "modal",
              onClick: ev => ev.stopPropagation()
            },
            e("h2", { className: "text-2xl font-bold mb-4" }, title),
            e("p", { className: "mb-4" }, message),
            e("button", { className: "px-4 py-2 bg-blue-600 text-white rounded", onClick: onClose }, "Close")
          )
        );
      }
      
      function createRipple(event) {
        const button = event.currentTarget;
        const ripple = document.createElement("span");
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;
        ripple.style.width = ripple.style.height = `${diameter}px`;
        ripple.style.left = `${event.clientX - button.offsetLeft - radius}px`;
        ripple.style.top = `${event.clientY - button.offsetTop - radius}px`;
        ripple.className = "ripple";
        button.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
      }
      
      function App() {
        const [selectedUser, setSelectedUser] = useState("default_user");
        const [currentTheme, setCurrentTheme] = useState("light-theme");
        const [modalOpen, setModalOpen] = useState(false);
        useEffect(() => {
          document.body.classList.remove("light-theme", "solarized-theme", "material-theme", "monokai-theme");
          document.body.classList.add(currentTheme);
        }, [currentTheme]);
        useEffect(() => {
          const timer = setTimeout(() => {
            setModalOpen(true);
          }, 2000);
          return () => clearTimeout(timer);
        }, []);
        return e(
          "div",
          { className: "flex flex-col h-screen" },
          e(
            "header",
            { className: "flex justify-between items-center px-4" },
            e("h1", { className: "text-2xl font-bold" }, "Advanced Diabetes Dashboard"),
            e(ThemeSwitcher, { currentTheme, onThemeChange: setCurrentTheme })
          ),
          e(CustomTabs, { selectedUser }),
          e(
            "main",
            null,
            e(
              Router,
              null,
              e(AllUsers, { path: "/all_users", onSelect: setSelectedUser }),
              e(ChatTab, { path: "/chat/:userId" }),
              e(ProfileTab, { path: "/profile/:userId" }),
              e(ChartsTab, { path: "/charts/:userId" }),
              e(RecsTab, { path: "/recs/:userId" }),
              e(KGTab, { path: "/kg" }),
              e(HealthDashboard, { path: "/health", userId: selectedUser }),
              e(IKADAssistant, { path: "/ikad" }),
              e(AllUsers, { path: "*", onSelect: setSelectedUser })
            )
          ),
          modalOpen && e(Modal, { title: "Welcome!", message: "This is an integrated modal using advanced animations.", onClose: () => setModalOpen(false) })
        );
      }
      
      render(e(App), document.getElementById("root"));
    </script>
  </body>
</html>

'''