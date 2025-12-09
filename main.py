import streamlit as st
import os
import tempfile
import pandas as pd
import re
import unicodedata
import nltk
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
# from db_utils import get_recent_assessments_text
from db import save_assessment_result, load_assessment_results, get_recent_assessments_text
from datetime import datetime

# Try different langchain imports based on version
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
except ImportError:
    # Fallback for older versions
    try:
        # from langchain.chat_models import ChatOpenAI
        # from langchain.embeddings import OpenAIEmbeddings
        #from langchain.vectorstores import FAISS
        from langchain_community.vectorstores import FAISS
        #from langchain.text_splitter import RecursiveCharacterTextSplitter
        # from langchain.schema import Document
        # from langchain.chains import RetrievalQA, ConversationalRetrievalChain
        # from langchain.prompts import ChatPromptTemplate
        # from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
        InMemoryVectorStore = FAISS
    except ImportError as e:
        st.error(f"LangChain import error: {e}")
        st.stop()

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    st.warning("pdfplumber not available. Install with: pip install pdfplumber")

nltk.download('punkt_tab')

# Data classes for learning system
@dataclass
class AssessmentQuestion:
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: str  # "basic", "intermediate", "advanced"
    subtopic:str

@dataclass
class AssessmentResult:
    topic: str
    score: float
    total_questions: int
    correct_answers: int
    difficulty_distribution: Dict[str, int]
    weaknesses: List[str]
    strengths: List[str]
    recommended_level: str
    subtopic_performance: Dict[str, Dict[str, int]] = None

@dataclass
class LearnerProfile:
    user_id: str
    topic_assessments: Dict[str, AssessmentResult]
    learning_preferences: Dict[str, any]
    current_level: str
    last_updated: str

import re
import streamlit as st

def render_response(text: str):
    """Render AI response with LaTeX support and robust delimiter correction."""
    if not text:
        return
    
    # --- Preserve code blocks first ---
    code_blocks = []
    def _save_code(m):
        code_blocks.append(m.group(0))
        return f"[[CODEBLOCK_{len(code_blocks)-1}]]"
    
    tmp = re.sub(r'```.*?```', _save_code, text, flags=re.DOTALL)
    tmp = re.sub(r'`[^`]+`', _save_code, tmp)
    
    # --- Step 1: Convert all bracket-style delimiters to dollar signs FIRST ---
    # This must happen before other processing
    tmp = re.sub(r'\\\[', '$$', tmp)
    tmp = re.sub(r'\\\]', '$$', tmp)
    tmp = re.sub(r'\\\(', '$', tmp)
    tmp = re.sub(r'\\\)', '$', tmp)
    
    # --- Step 2: Fix escaped dollar signs that should be delimiters ---
    tmp = re.sub(r'\\(\$)', r'\1', tmp)
    
    # --- Step 3: Normalize display math ($$...$$) ---
    # Handle cases where $$ might have spaces or be on separate lines
    tmp = re.sub(r'\$\$\s*', '$$', tmp)
    tmp = re.sub(r'\s*\$\$', '$$', tmp)
    
    # --- Step 4: Fix orphaned $$ that should be $ ---
    # Match $$ that appear inline without proper pairing
    def _fix_inline_display(match):
        content = match.group(0)
        # Count $$ occurrences
        double_count = content.count('$$')
        if double_count % 2 == 1:  # Odd number means orphaned
            # Convert last $$ to $
            content = content[::-1].replace('$$', '$', 1)[::-1]
        return content
    
    lines = tmp.split('\n')
    fixed_lines = []
    for line in lines:
        if '$$' in line and line.count('$$') % 2 == 1:
            # Try to detect if this is truly inline
            if not re.match(r'^\s*\$\$', line):  # Not starting with $$
                line = line.replace('$$', '$', 1)  # Replace first occurrence
        fixed_lines.append(line)
    tmp = '\n'.join(fixed_lines)
    
    # --- Step 5: Wrap common LaTeX commands that aren't in math mode ---
    # Protect already wrapped content
    protected = []
    def _protect(m):
        protected.append(m.group(0))
        return f"[[PROTECTED_{len(protected)-1}]]"
    
    tmp = re.sub(r'\$\$.*?\$\$', _protect, tmp, flags=re.DOTALL)
    tmp = re.sub(r'\$[^\$\n]+?\$', _protect, tmp)
    
    # Now wrap unwrapped LaTeX
    latex_commands = r'\\(?:begin|end|frac|sqrt|sum|int|cdot|times|alpha|beta|gamma|delta|theta|' \
                    r'lambda|mu|sigma|pi|Delta|Sigma|vec|hat|bar|mathbf|mathbb|mathcal|' \
                    r'text|left|right|det|neq|leq|geq|pm|infty)'
    
    # Wrap display-style environments
    tmp = re.sub(
        r'(\\begin\{(?:pmatrix|bmatrix|vmatrix|matrix|align|equation|cases)\}.*?\\end\{(?:pmatrix|bmatrix|vmatrix|matrix|align|equation|cases)\})',
        r'$$\1$$',
        tmp,
        flags=re.DOTALL
    )
    
    # Wrap inline math expressions
    def _wrap_inline_math(match):
        content = match.group(0)
        # Don't wrap if it's just a word
        if re.match(r'^\\[a-zA-Z]+$', content) and content not in ['\\det', '\\neq']:
            return content
        return f'${content}$'
    
    # Match LaTeX commands with their arguments
    tmp = re.sub(
        r'(?<![\$\\])(' + latex_commands + r'(?:\{[^}]*\}|\([^)]*\))*(?:\s*[+\-=]\s*(?:[a-zA-Z0-9\s]|' + latex_commands + r'|\{[^}]*\})*)*)',
        _wrap_inline_math,
        tmp
    )
    
    # Restore protected content
    for i, block in enumerate(protected):
        tmp = tmp.replace(f"[[PROTECTED_{i}]]", block)
    
    # --- Step 6: Clean up multiple consecutive dollar signs ---
    tmp = re.sub(r'\$\$\$+', '$$', tmp)
    tmp = re.sub(r'(?<!\$)\$(?!\$)\s*(?<!\$)\$(?!\$)', '$$', tmp)
    
    # --- Step 7: Fix spacing around inline math ---
    tmp = re.sub(r'([^\s\$])\$([^\$])', r'\1 $\2', tmp)
    tmp = re.sub(r'([^\$])\$([^\s\$])', r'\1$ \2', tmp)
    
    # --- Step 8: Restore code blocks ---
    for i, block in enumerate(code_blocks):
        tmp = tmp.replace(f"[[CODEBLOCK_{i}]]", block)
    
    # --- Step 9: Inject MathJax configuration ---
    st.markdown("""
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']],
            processEscapes: true,
            processEnvironments: true,
            packages: {'[+]': ['ams', 'newcommand', 'configmacros']}
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process'
        },
        startup: {
            pageReady: () => {
                return MathJax.startup.defaultPageReady().then(() => {
                    console.log('MathJax initial typesetting complete');
                });
            }
        }
    };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """, unsafe_allow_html=True)
    
    # --- Step 10: Render the processed text ---
    st.markdown(tmp, unsafe_allow_html=True)
    
    # Force MathJax to re-render
    st.markdown("""
    <script>
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
    }
    </script>
    """, unsafe_allow_html=True)

# Simple memory management class
class SimpleMemory:
    def __init__(self, key: str = "memory_messages"):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = []

    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to memory"""
        st.session_state[self.key].append({
            "type": "human",
            "content": question
        })
        st.session_state[self.key].append({
            "type": "ai", 
            "content": answer
        })

    def get_messages(self):
        """Get all messages"""
        return st.session_state[self.key] if self.key in st.session_state else []

    def clear(self):
        """Clear all memory"""
        st.session_state[self.key] = []

    def format_for_prompt(self):
        """Format ALL memory for inclusion in prompt"""
        messages = self.get_messages()
        if not messages:
            return ""
        
        formatted_history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                formatted_history.append(f"Student: {human_msg['content']}")
                formatted_history.append(f"Tutor: {ai_msg['content']}")
        
        return "\n".join(formatted_history)

class AdaptiveLearningSystem:
    """Handles assessment generation, evaluation, and personalized learning paths"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever


    def parse_topics_from_response(self, topics_response: str) -> List[str]:
        """Parse main top-level topics (numbered, bulleted, or standalone headings) from LLM response."""
        if not topics_response:
            return []

        topics = []
        lines = topics_response.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common numbering/bullet patterns
            if line[0].isdigit():
                # e.g., "1. Introduction to Linear Algebra:" → "Introduction to Linear Algebra"
                line = line.lstrip("0123456789. ").strip()
            elif line.startswith(("-", "*", "•")):
                line = line.lstrip("-*• ").strip()

            # Drop empty or too short after stripping
            if not line or len(line) <= 2:
                continue

            # Ignore if it's clearly just an intro/sentence
            if line.lower().startswith(("main topics", "themes", "concepts")):
                continue
            if line.endswith(".") and not line.endswith("..."):
                continue

            # Remove trailing colon (common in headings)
            if line.endswith(":"):
                line = line[:-1].strip()

            # Deduplicate
            if line and line not in topics:
                topics.append(line)

        return topics[:15]

    
    def generate_assessment(self, topic: str, document_context: str) -> List[AssessmentQuestion]:
        """Generate assessment questions for a specific topic"""
        
        # Get relevant context for the topic
        #relevant_docs = self.retriever.get_relevant_documents(f"questions about {topic} examples problems")

        query = f"questions about {topic} examples problems"

        if hasattr(self.retriever, "invoke"):
            relevant_docs = self.retriever.invoke(query)
        else:
            relevant_docs = self.retriever.get_relevant_documents(query)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:]])
        
        prompt = f"""
        You are an expert educator creating assessment questions. Based on the topic "{topic}" 
        , generate exactly 6 multiple choice questions that test 
        understanding at different difficulty levels. You can generate questions on your own or from the document context.
        
        Requirements:
        - Question 1: Basic/foundational level
        - Question 2: Basic/foundational level  
        - Question 3: Intermediate level
        - Question 4: Intermediate level
        - Question 5: Advanced/application level
        - Question 6: Advanced/application level
        - Each question should have 5 options (A, B, C, D, E)
        - Include clear explanations for correct answers
        
        Document context:
        {context}
        
        Return your response in the following JSON format:
        {{
            "questions": [
                {{
                    "question": "Question text here?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4", "E) I don't know"],
                    "correct_answer": 0,
                    "explanation": "Why this answer is correct...",
                    "difficulty": "basic",
                    "subtopic": "specific them/subtopic of the question"
                }},
                ...
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the content to extract JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            data = json.loads(content)
            questions = []
            
            for q_data in data["questions"]:
                questions.append(AssessmentQuestion(
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    explanation=q_data["explanation"],
                    difficulty=q_data.get("difficulty", "basic"),
                    subtopic=q_data["subtopic"]
                ))
            
            return questions
            
        except Exception as e:
            st.error(f"Error generating assessment: {e}")
            # Return fallback questions
    
    def evaluate_assessment(self, questions: List[AssessmentQuestion], 
                          answers: List[int], topic: str) -> AssessmentResult:
        """Evaluate assessment results and create learner profile"""
        
        correct_answers = sum(1 for i, q in enumerate(questions) 
                             if i < len(answers) and answers[i] == q.correct_answer)
        score = correct_answers / len(questions) if questions else 0
        
        # Analyze by difficulty (keep existing logic)
        difficulty_correct = {"basic": 0, "intermediate": 0, "advanced": 0}
        difficulty_total = {"basic": 0, "intermediate": 0, "advanced": 0}
        
        # NEW: Analyze by subtopic/theme
        subtopic_performance = {}  # {"subtopic": {"correct": int, "total": int}}
        
        for i, question in enumerate(questions):
            # Existing difficulty tracking
            difficulty_total[question.difficulty] += 1
            if i < len(answers) and answers[i] == question.correct_answer:
                difficulty_correct[question.difficulty] += 1
            
            # NEW: Subtopic tracking
            subtopic = getattr(question, 'subtopic', 'General concept')
            if subtopic not in subtopic_performance:
                subtopic_performance[subtopic] = {"correct": 0, "total": 0}
            
            subtopic_performance[subtopic]["total"] += 1
            if i < len(answers) and answers[i] == question.correct_answer:
                subtopic_performance[subtopic]["correct"] += 1
        
        # NEW: Determine topic-based strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for subtopic, performance in subtopic_performance.items():
            if performance["total"] > 0:
                success_rate = performance["correct"] / performance["total"]
                if success_rate >= 0.75:  # 75% or better = strength
                    strengths.append(subtopic)
                elif success_rate < 0.5:  # Less than 50% = weakness
                    weaknesses.append(subtopic)
        
        # If no specific strengths/weaknesses found, fall back to difficulty-based
        if not strengths and not weaknesses:
            for difficulty, correct in difficulty_correct.items():
                total = difficulty_total[difficulty]
                if total > 0:
                    rate = correct / total
                    if rate >= 0.8:
                        strengths.append(f"{difficulty.title()} level concepts")
                    elif rate < 0.5:
                        weaknesses.append(f"{difficulty.title()} level concepts")
        
        # Recommend level
        if score >= 0.8:
            recommended_level = "advanced"
        elif score >= 0.6:
            recommended_level = "intermediate"
        else:
            recommended_level = "basic"
        
        return AssessmentResult(
            topic=topic,
            score=score,
            total_questions=len(questions),
            correct_answers=correct_answers,
            difficulty_distribution=difficulty_correct,
            subtopic_performance=subtopic_performance,  # NEW: Add subtopic performance data
            weaknesses=weaknesses,
            strengths=strengths,
            recommended_level=recommended_level
        )
    
    def get_personalized_content(
        self, user_id: str, topic: str, learner_level: str, weaknesses: List[str]
    ) -> str:
        # Retrieve course materials
        #relevant_docs = self.retriever.get_relevant_documents(f"{topic} {learner_level} level")

        query = f"{topic} {learner_level} level"

        if hasattr(self.retriever, "invoke"):
            relevant_docs = self.retriever.invoke(query)
        else:
            relevant_docs = self.retriever.get_relevant_documents(query)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:]])

        # Pull recent learner history from DB
        db_context = get_recent_assessments_text(user_id, topic=topic, limit=5)

        weakness_focus = ", ".join(weaknesses) if weaknesses else "general understanding"

        prompt = f"""
        You are a personalized AI tutor. Use the student's past performance
        and current results to generate tailored learning content.

        Learner history:
        {db_context}

        Current profile:
        - Topic: {topic}
        - Current Level: {learner_level}
        - Areas to improve: {weakness_focus}

        Course material context:
        {context}

        Please provide:
        1. Key concept explanations based on history (suited to their level). You can be detailed in this.
        2. Some practice questions, with explanations and solutions.
        3. Next topics to focus on.
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return f"I'm here to help you learn {topic}! Let's start with your weak areas."


def clean_text(text):
    """Clean and normalize text extracted from PDFs"""
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Fix common LaTeX conversion issues
    text = text.replace('ﬁ', 'fi')  # Fix ligatures
    text = text.replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff')
    text = text.replace('ﬃ', 'ffi')
    text = text.replace('ﬄ', 'ffl')
    
    # Remove excessive line breaks but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def extract_text_with_pymupdf(file_path):
    """Extract text using PyMuPDF (fitz) - often better for LaTeX PDFs"""
    if not PYMUPDF_AVAILABLE:
        return None
    try:
        doc = fitz.open(file_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            cleaned_text = clean_text(text)
            if cleaned_text:  # Only add non-empty pages
                texts.append(Document(
                    page_content=cleaned_text,
                    metadata={"source": file_path, "page": page_num + 1}
                ))
        doc.close()
        return texts
    except Exception as e:
        st.error(f"PyMuPDF extraction failed: {str(e)}")
        return None

def extract_text_with_pdfplumber(file_path):
    """Extract text using pdfplumber - good for structured documents"""
    if not PDFPLUMBER_AVAILABLE:
        return None
    try:
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                cleaned_text = clean_text(text)
                if cleaned_text:  # Only add non-empty pages
                    texts.append(Document(
                        page_content=cleaned_text,
                        metadata={"source": file_path, "page": page_num + 1}
                    ))
        return texts
    except Exception as e:
        st.error(f"PDFplumber extraction failed: {str(e)}")
        return None

def extract_pdf_text(file_path, method="auto"):
    """Extract text from PDF using multiple methods as fallback"""
    
    if method == "auto":
        # Try PyMuPDF first (often best for LaTeX)
        if PYMUPDF_AVAILABLE:
            docs = extract_text_with_pymupdf(file_path)
            if docs and len(docs) > 0:
                # Check if extraction was successful by examining text quality
                sample_text = docs[0].page_content[:500] if docs else ""
                if len(sample_text) > 50 and not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using PyMuPDF")
                    return docs
        
        # Try pdfplumber as second option
        if PDFPLUMBER_AVAILABLE:
            st.warning("PyMuPDF didn't work well, trying pdfplumber...")
            docs = extract_text_with_pdfplumber(file_path)
            if docs and len(docs) > 0:
                sample_text = docs[0].page_content[:500] if docs else ""
                if len(sample_text) > 50 and not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using pdfplumber")
                    return docs
        
        # Fall back to PyPDFLoader
        st.warning("Trying PyPDFLoader as fallback...")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Clean the text from PyPDFLoader too
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            if docs and len(docs) > 0:
                sample_text = docs[0].page_content[:500] if docs else ""
                if not is_corrupted_text(sample_text):
                    st.success("✅ Successfully extracted text using PyPDFLoader")
                    return docs
        except Exception as e:
            st.error(f"PyPDFLoader failed: {str(e)}")
        
        # Try UnstructuredPDFLoader as last resort
        st.warning("Trying UnstructuredPDFLoader as last resort...")
        try:
            loader = UnstructuredPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            if docs and len(docs) > 0:
                st.success("✅ Successfully extracted text using UnstructuredPDFLoader")
                return docs
        except Exception as e:
            st.error(f"UnstructuredPDFLoader failed: {str(e)}")
    
    return None

def is_corrupted_text(text):
    """Check if text appears to be corrupted or encoded improperly"""
    if not text:
        return True
    
    # Check for excessive special characters
    special_char_ratio = sum(1 for c in text if not c.isprintable() and c not in '\n\t ') / len(text)
    if special_char_ratio > 0.3:  # More than 30% special characters
        return True
    
    # Check for patterns that suggest encoding issues
    corrupted_patterns = [
        r'[^\x00-\x7F]{10,}',  # Long sequences of non-ASCII
        r'[\x00-\x08\x0E-\x1F]{5,}',  # Control characters
        r'\?{5,}',  # Multiple question marks (encoding failures)
    ]
    
    for pattern in corrupted_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def format_chat_history_for_prompt(messages, max_exchanges=5):
    """Format simple message list for inclusion in prompt"""
    if not messages:
        return ""
    
    # Take only the last max_exchanges conversations
    recent_messages = messages[-max_exchanges*2:] if len(messages) > max_exchanges*2 else messages
    
    formatted_history = []
    for i in range(0, len(recent_messages), 2):
        if i + 1 < len(recent_messages):
            human_msg = recent_messages[i]
            ai_msg = recent_messages[i + 1]
            formatted_history.append(f"Student: {human_msg['content']}")
            formatted_history.append(f"Tutor: {ai_msg['content']}")
    
    return "\n".join(formatted_history)

def detect_problem_indicators(text):
    """Detect if text contains numerical problems or examples"""
    problem_indicators = [
        r'example\s+\d+',
        r'problem\s+\d+',
        r'exercise\s+\d+',
        r'question\s+\d+',
        r'solution:?',
        r'answer:?',
        r'calculate',
        r'find\s+the',
        r'solve\s+for',
        r'\$[0-9,]+',  # Dollar amounts
        r'\d+%',  # Percentages
        r'\d+\.\d+',  # Decimal numbers
        r'given:?',
        r'let\s+\w+\s*=',
        r'if\s+.*\s+then',
    ]
    
    score = 0
    for pattern in problem_indicators:
        matches = re.findall(pattern, text, re.IGNORECASE)
        score += len(matches)
    
    return score > 2  # Threshold for considering it a problem/example

def create_examples_retriever(docs, embeddings):
    """Create a specialized retriever for examples and problems"""
    # Filter documents that likely contain examples/problems
    example_docs = []
    for doc in docs:
        if detect_problem_indicators(doc.page_content):
            # Mark as example document
            doc.metadata['content_type'] = 'example'
            example_docs.append(doc)
    
    if example_docs:
        try:
            # examples_vectorstore = InMemoryVectorStore.from_documents(
            #     documents=example_docs, 
            #     embedding=embeddings
            # )
            examples_vectorstore = InMemoryVectorStore(embedding=embeddings)
            examples_vectorstore.add_documents(documents=example_docs)
        except:
            from langchain_community.vectorstores import FAISS
            examples_vectorstore = FAISS.from_documents(example_docs, embeddings)
        
        return examples_vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        return None

def create_conversational_chain(retriever, llm, examples_retriever=None):
    """Create a conversational retrieval chain with simple memory"""
    
    # Initialize memory
    memory = SimpleMemory("conversation_memory")
    
    # Enhanced system prompt with memory awareness and examples focus
    system_prompt = (
        "You are an AI tutor specializing in academic documents. "
        "Use the retrieved context and conversation history to provide helpful, educational responses. "
        "Your role is to:\n"
        "1. Answer questions based on the document context\n"
        "2. Build upon previous conversations to provide continuity\n"
        "3. Clarify concepts when the student seems confused\n"
        "4. Suggest related topics or follow-up questions\n"
        "5. Reference previous topics when relevant\n\n"
        "If mathematical formulas or technical terms appear garbled, infer meaning from context. "
        "If you don't know something, say so clearly. "
        "Keep responses educational and engaging.\n\n"
        "Previous conversation:\n{chat_history}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Current question: {question}\n\n"
        "Provide a comprehensive answer and suggest 2-3 relevant follow-up questions."
    )
    
    # Examples-focused system prompt
    examples_prompt = (
        "You are an AI tutor specialized in finding and explaining practical examples and numerical problems. "
        "Only provide practical examples and numerical problems when user asks questions on a specific mathematical topic or concept."
        "Don't output anything if the question is very general or just asks summary of the document.s"
        "Focus specifically on:\n"
        "1. Numerical problems with step-by-step solutions\n"
        "2. Practical examples that illustrate concepts\n"
        "3. Calculation methods and formulas\n"
        "4. Real-world applications\n"
        "5. Exercise problems and their solutions\n\n"
        "Based on the context below, provide concrete examples and numerical problems related to: {question}\n\n"
        "Context with examples:\n{examples_context}\n\n"
        "Provide specific numerical examples, step-by-step calculations, and practice problems if available."
    )
    
    def conversational_rag_chain(inputs):
        """Main conversational RAG chain"""
        question = inputs["question"]
        mode = inputs.get("mode", "general")  # general or examples
        
        if mode == "examples" and examples_retriever:
            # Retrieve example-focused documents
            #example_docs = examples_retriever.get_relevant_documents(question)

            # query = f"questions about {topic} examples problems"

            if hasattr(retriever, "invoke"):
                relevant_docs = retriever.invoke(question)
            else:
                relevant_docs = retriever.get_relevant_documents(question)
            
            examples_context = "\n\n".join([doc.page_content for doc in example_docs])
            
            # Format the examples-focused prompt
            full_prompt = examples_prompt.format(
                examples_context=examples_context,
                question=question
            )
            
            source_docs = example_docs
            
        else:
            # Standard retrieval for general questions
            # docs = retriever.get_relevant_documents(question)

            if hasattr(retriever, "invoke"):
                relevant_docs = retriever.invoke(question)
            else:
                relevant_docs = retriever.get_relevant_documents(question)
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get chat history
            chat_history = memory.format_for_prompt()
            
            # Format the prompt
            full_prompt = system_prompt.format(
                chat_history=chat_history,
                context=context,
                question=question
            )
            
            source_docs = docs
        
        # Get response from LLM
        try:
            # Try newer message format first
            messages = [{"role": "user", "content": full_prompt}]
            response = llm.invoke(messages)
            answer = response.content
        except:
            # Fallback to older format
            try:
                response = llm.invoke(full_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except:
                # Last resort - direct call
                answer = llm(full_prompt)
        
        # Store in memory only for general mode
        if mode == "general":
            memory.add_exchange(question, answer)
        
        return {
            "answer": answer,
            "source_documents": source_docs,
            "chat_history": memory.format_for_prompt() if mode == "general" else "",
            "mode": mode
        }
    
    return conversational_rag_chain

# Streamlit app title
st.title("PAROLE - Personalized AI for Reliable On-demand Learning")
st.caption("Personalized learning with assessments and tailored content!")

# Load API key securely from Streamlit Secrets
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🚨 OPENAI_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "examples_history" not in st.session_state:
    st.session_state.examples_history = []

if "memory_messages" not in st.session_state:
    st.session_state.memory_messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "examples_retriever" not in st.session_state:
    st.session_state.examples_retriever = None

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# New session state for adaptive learning
if "learning_system" not in st.session_state:
    st.session_state.learning_system = None

if "available_topics" not in st.session_state:
    st.session_state.available_topics = []

if "learner_profile" not in st.session_state:
    st.session_state.learner_profile = LearnerProfile(
        user_id="default_user",
        topic_assessments={},
        learning_preferences={},
        current_level="basic",
        last_updated=datetime.now().isoformat()
    )

if "current_assessment" not in st.session_state:
    st.session_state.current_assessment = None

if "assessment_answers" not in st.session_state:
    st.session_state.assessment_answers = []

if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

# Check for a new file upload
if uploaded_file:
    if uploaded_file != st.session_state.uploaded_file:
        # Reset session state for a new file
        st.session_state.uploaded_file = uploaded_file
        st.session_state.chat_history = []
        st.session_state.examples_history = []
        if "conversation_memory" in st.session_state:
            st.session_state.conversation_memory = []  # Clear memory for new document
        st.session_state.retrieval_chain = None
        st.session_state.examples_retriever = None
        st.session_state.current_query = ""
        st.session_state.topics_answer = None  # reset topics
        
        # Reset adaptive learning state
        st.session_state.available_topics = []
        st.session_state.current_assessment = None
        st.session_state.assessment_answers = []
        st.session_state.assessment_complete = False
        st.session_state.selected_topic = None
        
        st.write("Processing new file...")

        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Use enhanced PDF extraction
            docs = extract_pdf_text(temp_file_path, method="auto")
            
            if not docs:
                st.error("❌ Failed to extract readable text from the PDF. The file might be:")
                st.write("- Image-based (scanned) PDF without OCR")
                st.write("- Heavily encrypted or protected")
                st.write("- Corrupted or in an unsupported format")
                st.write("Please try converting the PDF to a different format or use OCR.")
                st.stop()
            
            # Remove empty documents
            docs = [doc for doc in docs if doc.page_content.strip()]
            
            if not docs:
                st.error("❌ No readable text found in the PDF.")
                st.stop()
                
            # Show content preview
            content = docs[0].page_content[:1000]  # Show more content for preview
            st.write(f"### Document preview (first 1000 characters):")
            st.text_area("Extracted text:", content, height=200, disabled=True)
            
            # Show extraction statistics
            total_pages = len(docs)
            total_chars = sum(len(doc.page_content) for doc in docs)
            
            # Count example-containing pages
            example_pages = sum(1 for doc in docs if detect_problem_indicators(doc.page_content))
            
            st.info(f"📊 Successfully processed {total_pages} pages with {total_chars:,} characters")
            st.info(f"🔢 Found {example_pages} pages containing examples/problems")

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            combined_text = df.to_string(index=False)
            docs = [Document(page_content=combined_text, metadata={"source": uploaded_file.name})]
            content = docs[0].page_content[:1000]
            st.write(f"### Document preview (first 1000 characters):")
            st.text_area("Excel content:", content, height=200, disabled=True)
        else:
            st.error("Unsupported file type. Please upload a PDF or Excel file.")
            st.stop()

        # Save docs for later use
        st.session_state.docs = docs

        # Preprocess and split text with better chunking for LaTeX documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for academic documents
            chunk_overlap=300,  # More overlap to preserve context
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for academic text
        )
        splits = text_splitter.split_documents(docs)
        
        st.info(f"📄 Created {len(splits)} text chunks for processing")

        # Create vector store
        try:
            # Initialize embeddings first
            embeddings = OpenAIEmbeddings()
            
            # Create main vector store
            try:
                vectorstore = InMemoryVectorStore.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
            except Exception as e:
                # Fallback for older versions
                st.warning(f"InMemoryVectorStore failed, trying FAISS: {e}")
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.from_documents(splits, embeddings)
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Create examples retriever
            examples_retriever = create_examples_retriever(splits, embeddings)
            st.session_state.examples_retriever = examples_retriever
            
            if examples_retriever:
                st.success("✅ Created specialized examples retriever!")
            else:
                st.warning("⚠️ No numerical examples detected in this document")

            # Set up the LLM
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            except Exception as e:
                st.warning(f"Failed to initialize gpt-4o, trying gpt-3.5-turbo: {e}")
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
            # Save llm for later reuse
            st.session_state.llm = llm

            # Create conversational chain with memory and examples
            st.session_state.retrieval_chain = create_conversational_chain(
                retriever, llm, examples_retriever
            )
            
            # Initialize adaptive learning system
            st.session_state.learning_system = AdaptiveLearningSystem(llm, retriever)
            
            st.success("✅ Document processed successfully! Ready for questions with examples & memory!")

            # === Auto-extract topics for adaptive learning ===
            try:
                all_text = "\n".join([doc.page_content for doc in docs])
                
                # Extract traditional topics for the Topics tab
                prompt = (
                    "You are an expert academic assistant. "
                    "Given the following document content, identify the main topics, "
                    "themes, and concepts covered in the text. "
                    "Return them as a clear bullet-point list, grouped logically if possible.\n\n"
                    f"{all_text}"  # truncate for safety
                )
                response = llm.invoke(prompt)
                st.session_state.topics_answer = getattr(response, "content", str(response))
                
                # # Extract learning topics for adaptive learning
                # learning_topics = st.session_state.learning_system.extract_learning_topics(all_text)
                # st.session_state.available_topics = learning_topics

                topics_response = st.session_state.topics_answer
                # Parse the same topics for use in adaptive learning (Tab 4)
                parsed_topics = st.session_state.learning_system.parse_topics_from_response(topics_response)
                st.session_state.available_topics = parsed_topics
                
                st.success("✅ Extracted main topics and learning topics from the document!")
                # st.info(f"🎯 Found {len(learning_topics)} specific learning topics for adaptive learning")
                
            except Exception as e:
                st.error(f"❌ Error extracting topics automatically: {str(e)}")
                st.session_state.topics_answer = None
                #st.session_state.available_topics = ["Basic Concepts", "Intermediate Topics", "Advanced Applications"]
            
        except Exception as e:
            st.error(f"❌ Error creating retrieval chain: {str(e)}")
            st.error("This might be due to LangChain version incompatibility. Please try:")
            st.code("pip install langchain==0.0.352 langchain-openai==0.0.5")
            st.stop()

# Query interface with tabs
if st.session_state.retrieval_chain:
    
    # Create tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(["💬 General Q&A", "🔢 Examples & Problems", "📑 Document Topics", "🎯 Adaptive Learning"])
    
    with tab1:
        st.write("### General Questions & Conceptual Understanding")
        
        # Show conversation context if available
        if "conversation_memory" in st.session_state and st.session_state.conversation_memory:
            with st.expander("💭 Full Conversation Context", expanded=False):
                full_context = format_chat_history_for_prompt(st.session_state.conversation_memory)
                st.text_area("Complete Conversation History", full_context, height=300)
        
        query = st.text_input("Ask your question:", placeholder="I'll remember our conversation context...", key="general_query")
        submit_query = st.button("Submit Question", key="general_submit")

        if submit_query and query.strip():
            with st.spinner("Thinking with context..."):
                try:
                    # 1. General answer
                    result_general = st.session_state.retrieval_chain({"question": query, "mode": "general"})
                    answer_general = result_general.get("answer", "No answer available.")
                    contexts_general = result_general.get("source_documents", [])
                    
                    st.session_state.chat_history.append({
                        "question": query, 
                        "answer": answer_general
                    })
                    
                    st.success("Answer:")
                    render_response(answer_general)

                    # Show source pages for general mode
                    if uploaded_file and uploaded_file.type == "application/pdf" and contexts_general:
                        pages = [ctx.metadata.get("page") for ctx in contexts_general if "page" in ctx.metadata]
                        if pages:
                            st.write("#### Answer retrieved from pages:")
                            st.write(", ".join([f"Page {p}" for p in sorted(set(pages))]))

                    # 2. Auto-trigger examples mode
                    if st.session_state.examples_retriever:
                        result_examples = st.session_state.retrieval_chain({"question": query, "mode": "examples"})
                        answer_examples = result_examples.get("answer", "No examples found.")
                        contexts_examples = result_examples.get("source_documents", [])

                        st.session_state.examples_history.append({
                            "question": query, 
                            "answer": answer_examples
                        })

                        # Show in tab1 too (optional)
                        if answer_examples.strip() and len(answer_examples) > 50:  # Only show if meaningful content
                            st.info("🔢 Related Examples & Problems:")
                            render_response(answer_examples)

                            if uploaded_file and uploaded_file.type == "application/pdf" and contexts_examples:
                                pages_ex = [ctx.metadata.get("page") for ctx in contexts_examples if "page" in ctx.metadata]
                                if pages_ex:
                                    st.write("#### Examples found on pages:")
                                    st.write(", ".join([f"Page {p}" for p in sorted(set(pages_ex))]))

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

    with tab2:
        st.write("### Numerical Problems & Practical Examples")
        
        if not st.session_state.examples_retriever:
            st.warning("⚠️ No numerical examples detected in this document")
            st.write("This section works best with textbooks containing:")
            st.write("- Numbered examples or problems")
            st.write("- Step-by-step solutions")
            st.write("- Numerical calculations")
            st.write("- Practice exercises")
        else:
            st.info("🎯 This mode focuses specifically on finding numerical problems and practical examples from your document")
        
        examples_query = st.text_input(
            "Ask for examples or problems:", 
            placeholder="Show me examples of... / Find problems about... / Give me practice questions on...", 
            key="examples_query"
        )
        submit_examples = st.button("Find Examples", key="examples_submit")
        
        if submit_examples and examples_query.strip():
            with st.spinner("Searching for examples and problems..."):
                try:
                    result = st.session_state.retrieval_chain({
                        "question": examples_query, 
                        "mode": "examples"
                    })
                    
                    answer = result.get("answer", "No examples found.")
                    contexts = result.get("source_documents", [])
                    
                    st.session_state.examples_history.append({
                        "question": examples_query, 
                        "answer": answer
                    })
                    
                    st.success("📚 Examples & Problems Found:")
                    render_response(answer)
                    
                    # Show source pages for examples
                    if uploaded_file and uploaded_file.type == "application/pdf" and contexts:
                        pages = []
                        for context in contexts:
                            metadata = getattr(context, 'metadata', {})
                            if 'page' in metadata:
                                pages.append(metadata['page'])
                        
                        if pages:
                            unique_pages = sorted(set(pages))
                            st.write("#### Examples found on pages:")
                            st.write(", ".join([f"Page {page}" for page in unique_pages]))
                            
                            # Show example content preview
                            with st.expander("📖 Preview Example Content"):
                                for context in contexts[:2]:  # Show first 2 contexts
                                    page_num = getattr(context, 'metadata', {}).get('page', 'Unknown')
                                    content = getattr(context, 'page_content', '')[:800]
                                    st.write(f"**Page {page_num}:**")
                                    st.text_area(f"Content from page {page_num}", content, height=150, key=f"preview_{page_num}")
                                
                except Exception as e:
                    st.error(f"❌ Error finding examples: {str(e)}")
    
    with tab3:
        st.write("### 📑 Main Topics in the Document")
        
        if "topics_answer" in st.session_state and st.session_state.topics_answer:
            render_response(st.session_state.topics_answer)
        else:
            st.info("📂 Please upload a document to extract topics.")

    # with tab4:
    #     st.write("### 🎯 Adaptive Learning System")
        
    #     if not st.session_state.available_topics:
    #         st.info("📚 Please upload a document first to enable adaptive learning.")
    with tab4:
        st.write("### 🎯 Adaptive Learning System")
        
        if not st.session_state.available_topics:
            st.info("📚 Please upload a document first to enable adaptive learning.")
            if "topics_answer" in st.session_state and st.session_state.topics_answer:
                st.warning("Topics were found in the document but couldn't be parsed for assessment. Please check the document content or try re-uploading.")
        else:
            # Display learner profile summary
            profile = st.session_state.learner_profile
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Level", profile.current_level.title())
            with col2:
                st.metric("Topics Assessed", len(profile.topic_assessments))
            with col3:
                last_updated = datetime.fromisoformat(profile.last_updated).strftime("%m/%d %H:%M")
                st.metric("Last Updated", last_updated)
            
            # Topic selection for assessment
            st.write("#### 📋 Choose a Topic to Learn")
            
            if st.session_state.available_topics:
                selected_topic = st.selectbox(
                    "Select a topic for assessment and personalized learning:",
                    options=[""] + st.session_state.available_topics,
                    key="topic_selector"
                )
                
                if selected_topic and selected_topic != st.session_state.selected_topic:
                    st.session_state.selected_topic = selected_topic
                    st.session_state.current_assessment = None
                    st.session_state.assessment_answers = []
                    st.session_state.assessment_complete = False
                
                # Start Assessment Button
                if selected_topic and st.button("🧪 Start Assessment", key="start_assessment"):
                    with st.spinner(f"Generating assessment for {selected_topic}..."):
                        try:
                            # Get document context for the topic
                            docs = st.session_state.docs
                            document_content = "\n".join([doc.page_content for doc in docs])
                            
                            # Generate assessment
                            questions = st.session_state.learning_system.generate_assessment(
                                selected_topic, document_content
                            )
                            
                            st.session_state.current_assessment = questions
                            st.session_state.assessment_answers = []
                            st.session_state.assessment_complete = False
                            
                            st.success(f"✅ Assessment ready for {selected_topic}!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error generating assessment: {str(e)}")
            
            # Display Assessment
            if st.session_state.current_assessment and not st.session_state.assessment_complete:
                st.write("#### 📝 Assessment Questions")
                st.info(f"Topic: **{st.session_state.selected_topic}**")
                
                questions = st.session_state.current_assessment
                answers = []
                
                for i, question in enumerate(questions):
                    st.write(f"**Question {i+1}:** {question.question}")
                    
                    # Display difficulty level
                    difficulty_colors = {
                        "basic": "🟢",
                        "intermediate": "🟡", 
                        "advanced": "🔴"
                    }
                    st.caption(f"{difficulty_colors.get(question.difficulty, '⚪')} Difficulty: {question.difficulty.title()}")
                    
                    answer = st.radio(
                        f"Select your answer for Question {i+1}:",
                        options=range(len(question.options)),
                        format_func=lambda x, opts=question.options: opts[x],
                        key=f"q_{i}"
                    )
                    answers.append(answer)

                if st.button("📊 Submit Assessment", key="submit_assessment"):
                    result = st.session_state.learning_system.evaluate_assessment(
                        questions, answers, st.session_state.selected_topic
                    )

                    # Save in DB
                    save_assessment_result(
                        user_id=profile.user_id,
                        topic=st.session_state.selected_topic,
                        assessment_result=result,
                        questions=questions,
                        answers=answers
                    )

                    # Update session state
                    st.session_state.assessment_answers = answers
                    st.session_state.assessment_complete = True

                    # Update learner profile
                    st.session_state.learner_profile.topic_assessments[st.session_state.selected_topic] = result
                    st.session_state.learner_profile.current_level = result.recommended_level
                    st.session_state.learner_profile.last_updated = datetime.now().isoformat()

                    st.success("✅ Assessment completed! Scroll down to see your results.")
                            
            # Display Assessment Results and Personalized Content
            if st.session_state.assessment_complete and st.session_state.selected_topic:
                topic = st.session_state.selected_topic
                if topic in st.session_state.learner_profile.topic_assessments:
                    result = st.session_state.learner_profile.topic_assessments[topic]
                    
                    st.write("#### 📊 Assessment Results")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score", f"{result.score:.1%}")
                    with col2:
                        st.metric("Correct Answers", f"{result.correct_answers}/{result.total_questions}")
                    with col3:
                        st.metric("Recommended Level", result.recommended_level.title())
                    
                    # # Detailed breakdown
                    # if result.strengths:
                    #     st.success(f"**Strengths:** {', '.join(result.strengths)}")
                    if result.weaknesses:
                        st.warning(f"**Areas to improve:** {', '.join(result.weaknesses)}")
                    
                    # Show correct answers and explanations
                    with st.expander("📋 Review Questions and Answers"):
                        questions = st.session_state.current_assessment
                        user_answers = st.session_state.assessment_answers
                        
                        for i, (question, user_answer) in enumerate(zip(questions, user_answers)):
                            correct = user_answer == question.correct_answer
                            st.write(f"**Question {i+1}:** {question.question}")
                            
                            if correct:
                                st.success(f"✅ Your answer: {question.options[user_answer]}")
                            else:
                                st.error(f"❌ Your answer: {question.options[user_answer]}")
                                st.info(f"✅ Correct answer: {question.options[question.correct_answer]}")
                            
                            st.write(f"**Explanation:** {question.explanation}")
                            st.markdown("---")
                    
                    # Personalized Learning Content
                    st.write("#### 🎯 Personalized Learning Plan")
                    
                    with st.spinner("Generating personalized content..."):
                        try:
                            personalized_content = st.session_state.learning_system.get_personalized_content(
                                user_id=profile.user_id, topic=topic, learner_level=result.recommended_level, 
                                weaknesses=result.weaknesses
                            )
                            
                            render_response(personalized_content)
                            
                        except Exception as e:
                            st.error(f"❌ Error generating personalized content: {str(e)}")
                    
                    # Reset button
                    if st.button("🔄 Take Another Assessment", key="reset_assessment"):
                        st.session_state.current_assessment = None
                        st.session_state.assessment_answers = []
                        st.session_state.assessment_complete = False
                        st.session_state.selected_topic = None
                        st.rerun()
            
            # Learning History
            if st.session_state.learner_profile.topic_assessments:
                st.write("#### 📈 Learning Progress")
                
                with st.expander("📊 Assessment History", expanded=False):
                    for topic, assessment in st.session_state.learner_profile.topic_assessments.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**{topic}**")
                        with col2:
                            st.write(f"Score: {assessment.score:.1%}")
                        with col3:
                            st.write(f"Level: {assessment.recommended_level.title()}")

# Display chat histories
col1, col2 = st.columns(2)

with col1:
    if st.session_state.chat_history:
        st.write("### 💬 General Q&A History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                st.write(f"**Question:** {chat['question']}")
                render_response(chat['answer'])

with col2:
    if st.session_state.examples_history:
        st.write("### 🔢 Examples History")
        for i, example in enumerate(reversed(st.session_state.examples_history[-5:])):  # Show last 5
            with st.expander(f"Ex{len(st.session_state.examples_history) - i}: {example['question'][:50]}..."):
                st.write(f"**Query:** {example['question']}")
                render_response(example['answer'])

# Memory visualization - show all conversation history
if "conversation_memory" in st.session_state and st.session_state.conversation_memory and st.checkbox("Show Full Conversation History"):
    st.write("### 💬 Complete Conversation History")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", len(st.session_state.conversation_memory))
    with col2:
        st.metric("Conversation Turns", len(st.session_state.conversation_memory) // 2)
    
    # Show ALL memory as a timeline
    st.write("**Complete Conversation Timeline:**")
    for i, msg in enumerate(st.session_state.conversation_memory):
        role = "🧑‍🎓 Student" if msg['type'] == 'human' else "🤖 Tutor"
        with st.container():
            st.write(f"**{i+1}.** {role}: {msg['content'][:400]}...")
            st.markdown("---")

# Clear memory button
if st.sidebar.button("🗑️ Clear All Memory"):
    if "conversation_memory" in st.session_state:
        st.session_state.conversation_memory = []
    st.session_state.chat_history = []
    st.session_state.examples_history = []
    st.success("✅ All conversation memory cleared!")

# Clear learning progress button
if st.sidebar.button("🎯 Reset Learning Progress"):
    st.session_state.learner_profile = LearnerProfile(
        user_id="default_user",
        topic_assessments={},
        learning_preferences={},
        current_level="basic",
        last_updated=datetime.now().isoformat()
    )
    st.session_state.current_assessment = None
    st.session_state.assessment_answers = []
    st.session_state.assessment_complete = False
    st.session_state.selected_topic = None
    st.success("✅ Learning progress reset!")
