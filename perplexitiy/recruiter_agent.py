import smtplib
import re
from logger import get_logger
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

sys.stdout.reconfigure(encoding="utf-8")

class Email_schema(BaseModel):
    subject: str = Field(description="Subject of mail")
    body: str = Field(description="Body of mail")

logger = get_logger(__name__)

class Recruiter_agent:
    def __init__(self, llm, parser):
        self.llm = llm 
        self.parser = parser
        self.format_instructions = self.parser.get_format_instructions()

    def mail_sender(self, query: str, receiver_mail: str):
        """Generate and send a professional email based on the query to the specified receiver."""
        
        load_dotenv()

        app_pass = os.getenv("APP_PASSWORD")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
        if not GROQ_API_KEY:
            logger.error("❌ GROQ_API_KEY not found in .env file")
            raise ValueError("GROQ_API_KEY is required")

        if not app_pass:
            logger.error("❌ APP_PASSWORD not found in .env file")
            raise ValueError("APP_PASSWORD is required for Gmail SMTP")
        
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            if not receiver_mail or not receiver_mail.strip():
                raise ValueError("Receiver email address is required")
            if "@" not in receiver_mail:
                raise ValueError("Invalid email address format")
            
            sender_email = "gameranurag24@gmail.com"
            logger.info(f"Generating email for query: {query}")
            logger.info(f"Sending email to: {receiver_mail}")

            logger.info("Invoking LLM to generate email...")
            # Just invoke the LLM without parser
            response = self.llm.invoke(query)
            response_text = response.content

            # Extract subject and body from the plain text response
            subject = self._extract_subject(response_text)
            body = self._extract_body(response_text)

            # Create email
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = receiver_mail
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # SMTP Server
            logger.info(f"Connecting to Gmail SMTP server...")
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                logger.info("Starting TLS...")
                server.starttls()
                logger.info(f"Logging in as {sender_email}...")
                server.login(sender_email, app_pass)
                logger.info(f"Sending email to {receiver_mail}...")
                server.send_message(msg)
                logger.info("✅ Email sent successfully")

            return f"""
    ✅ Email Sent Successfully

    To: {receiver_mail}

    Subject:
    {subject}

    Body:
    {body}
    """

        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _extract_subject(self, text: str) -> str:
        """Extract subject from email text using multiple patterns."""
        # Try multiple patterns to find subject
        patterns = [
            r'\*\*?Subject:\s*(.+?)(?:\*\*|$)',  # **Subject: text** or **Subject: text
            r'^Subject:\s*(.+?)$',  # Subject: text (start of line)
            r'Application for\s+(.+?)(?:\s+Position)?$',  # Application for X Position
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                # Clean up markdown
                subject = re.sub(r'\*\*', '', subject)
                subject = re.sub(r'^[-*]+\s*', '', subject)  # Remove leading dashes or asterisks
                if subject and len(subject) > 3:
                    return subject
        
        # Default fallback
        return "Professional Email - Job Application"
    
    def _extract_body(self, text: str) -> str:
        """Extract body from email text, removing extra formatting."""
        # Remove markdown bold formatting
        body = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # Remove HTML-like tags if any
        body = re.sub(r'<[^>]+>', '', body)
        # Clean up excessive newlines
        body = re.sub(r'\n\n\n+', '\n\n', body)
        # Remove leading/trailing whitespace on each line but preserve structure
        lines = [line.rstrip() for line in body.split('\n')]
        body = '\n'.join(lines).strip()
        return body

        
def recruit_agent(query, receiver_mail):
    """ Recruit Agent to send a professional mail to the recruiter """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not found in .env file")
        raise ValueError("GROQ_API_KEY is required")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    parser = JsonOutputParser(pydantic_object=Email_schema)
    recruiter = Recruiter_agent(llm, parser)
    mail = recruiter.mail_sender(query, receiver_mail)
    return mail

def processing(df):
    """
    Process dataframe records and send personalized emails to recruiters
    
    Args:
        df: DataFrame with columns 'name', 'email', 'job_description'
    """
    
    for i in range(len(df)):
        name = df.loc[i, "name"]
        email = df.loc[i, "email"]
        job_description = df.loc[i, "job_description"]
        
        PROMPT= f"""You are a professional mail writer with over 10+ years of experience in sending emails to recruiters with a 100% success rate.
My name is Anurag Kumar, Mob Number- 9999999999
Generate a professional email for the following:

Hiring person Name: {name}
Job Description: {job_description}

Instructions:
- Tailor the email to match the job description
- Highlight these relevant skills: Python, Machine Learning, Artificial Intelligence, PyTorch, TensorFlow, Pandas, Numpy
- Create a compelling subject line
- Write a professional, well-formatted body
- Make it personalized and engaging

Skills to highlight:
- Python
- Machine Learning
- Artificial Intelligence
- PyTorch
- TensorFlow
- Pandas and Numpy"""
        
        try:
            recruit_agent(PROMPT, email)
            logger.info(f"✅ Email sent successfully to {email}")
        except Exception as e:
            logger.error(f"❌ Failed to send email to {email}: {str(e)}")
    
    logger.info("All email processing completed")