import imaplib
import email
import time
import schedule
import logging
import pandas as pd
from datetime import datetime
import re
import joblib
from email.header import decode_header


class EmailScraper:
    def __init__(self, email_address, password, imap_server='imap.gmail.com'):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.mail = None
        self.processed_emails = set()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('email_scraping.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.load_models()

    def load_models(self):
        try:
            # UPDATED paths
            self.detection_model, self.detection_vectorizer = joblib.load('trained_model.pkl')
            self.intent_model, self.intent_vectorizer = joblib.load('intent_classifier.pkl')
            self.logger.info("ML models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.detection_model = None
            self.intent_model = None

    def connect_to_mailbox(self):
        try:
            self.mail = imaplib.IMAP4_SSL(self.imap_server)
            self.mail.login(self.email_address, self.password)
            self.mail.select('INBOX')
            self.logger.info("Connected to mailbox successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to mailbox: {e}")
            return False

    def clean_text(self, text):
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = re.sub(r' +', ' ', text)
        return text.lower().strip()

    def predict_email(self, subject, body):
        if not self.detection_model or not self.intent_model:
            return "Models not loaded"
        combined_text = f"{subject or ''} {body or ''}"
        cleaned_text = self.clean_text(combined_text)
        try:
            X_main = self.detection_vectorizer.transform([cleaned_text])
            phishing_pred = self.detection_model.predict(X_main)[0]
            if phishing_pred == 1:
                X_intent = self.intent_vectorizer.transform([cleaned_text])
                intent_pred = self.intent_model.predict(X_intent)[0]
                return f"Phishing Detected! Intent: {intent_pred}"
            else:
                return "Safe Email (Ham)"
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "Prediction Error"

    def decode_email_subject(self, subject):
        if subject:
            decoded_subject = decode_header(subject)
            subject_text = ""
            for part, encoding in decoded_subject:
                if isinstance(part, bytes):
                    subject_text += part.decode(encoding or 'utf-8')
                else:
                    subject_text += part
            return subject_text
        return ""

    def get_email_body(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    body = re.sub(r'<[^>]+>', '', html_body)
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        return body

    def fetch_new_emails(self):
        try:
            if not self.mail:
                if not self.connect_to_mailbox():
                    return []
            typ, messages = self.mail.search(None, 'UNSEEN')
            email_ids = messages[0].split()
            processed_emails = []
            for email_id in email_ids:
                if email_id.decode() in self.processed_emails:
                    continue
                try:
                    typ, msg_data = self.mail.fetch(email_id, '(RFC822)')
                    email_message = email.message_from_bytes(msg_data[0][1])
                    subject = self.decode_email_subject(email_message["Subject"])
                    sender = email_message["From"]
                    date_received = email_message["Date"]
                    body = self.get_email_body(email_message)
                    prediction = self.predict_email(subject, body)
                    email_data = {
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'date': date_received,
                        'body_preview': body[:200] + "..." if len(body) > 200 else body,
                        'prediction': prediction,
                        'timestamp': datetime.now().isoformat()
                    }
                    processed_emails.append(email_data)
                    self.processed_emails.add(email_id.decode())
                    self.logger.info(f"Processed email from {sender}: {prediction}")
                    self.save_to_csv(email_data)
                except Exception as e:
                    self.logger.error(f"Error processing email {email_id}: {e}")
                    continue
            return processed_emails
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
            return []

    def save_to_csv(self, email_data):
        try:
            df = pd.DataFrame([email_data])
            try:
                existing_df = pd.read_csv('processed_emails.csv')
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass
            df.to_csv('processed_emails.csv', index=False)
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")

    def run_continuous_monitoring(self):
        self.logger.info("Starting continuous email monitoring...")
        while True:
            try:
                new_emails = self.fetch_new_emails()
                if new_emails:
                    self.logger.info(f"Processed {len(new_emails)} new emails")
                time.sleep(30)
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

    def scheduled_check(self):
        new_emails = self.fetch_new_emails()
        if new_emails:
            self.logger.info(f"Scheduled check processed {len(new_emails)} emails")

    def close_connection(self):
        if self.mail:
            self.mail.logout()

def main():
    EMAIL_ADDRESS = "drivetvmovie123@gmail.com"
    APP_PASSWORD = "lxbfrgengigujpwf"
    scraper = EmailScraper(EMAIL_ADDRESS, APP_PASSWORD)
    # Option 1: Continuous monitoring (uncomment to use)
    # scraper.run_continuous_monitoring()
    # Option 2: Scheduled monitoring (example: every 5 minutes)
    schedule.every(5).minutes.do(scraper.scheduled_check)
    print("Email monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping email monitoring...")
        scraper.close_connection()

if __name__ == "__main__":
    main()
