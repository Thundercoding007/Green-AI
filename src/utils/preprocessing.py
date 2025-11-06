# File: src/utils/preprocessing.py
# Email Data Preprocessing Utilities (Enhanced + Safe + Organized)

import re
import email
import unicodedata
import pandas as pd
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
from typing import List, Tuple
from tqdm import tqdm


class EmailPreprocessor:
    """Handles email cleaning, parsing, and anonymization"""

    def __init__(self):
        # Precompile regex patterns for efficiency
        self.email_pattern = re.compile(r'\S+@\S+')
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]'
            r'|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

    # -----------------------------------------------------------------
    def clean_text(self, text: str, anonymize: bool = True) -> str:
        """
        Clean and preprocess email text.
        Removes HTML, normalizes Unicode, strips noise, and anonymizes PII.

        Args:
            text: Raw email text
            anonymize: Whether to remove PII (emails, phones, URLs)

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML safely
        text = self.remove_html(text)

        # Normalize Unicode
        text = unicodedata.normalize('NFKD', text)

        # Optionally anonymize personal info
        if anonymize:
            text = self.anonymize_pii(text)

        # Keep only meaningful punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\']', ' ', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # -----------------------------------------------------------------
    def remove_html(self, text: str) -> str:
        """Remove HTML tags safely (falls back if BeautifulSoup fails)."""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(separator=' ')
        except Exception:
            # Fallback: simple regex-based tag removal
            return re.sub(r'<[^>]+>', ' ', text)

    # -----------------------------------------------------------------
    def anonymize_pii(self, text: str) -> str:
        """Anonymize personally identifiable information."""
        text = self.email_pattern.sub('[EMAIL]', text)
        text = self.url_pattern.sub('[URL]', text)
        text = self.phone_pattern.sub('[PHONE]', text)
        return text

    # -----------------------------------------------------------------
    def extract_email_parts(self, raw_email: str) -> dict:
        """
        Extract structured parts (subject, body, metadata) from raw email text.
        """
        try:
            msg = email.message_from_string(raw_email, policy=policy.default)
            subject = msg.get('subject', '') or ""
            body = ""

            # Extract text/plain and text/html bodies
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body += part.get_content()
                    elif content_type == "text/html":
                        body += self.remove_html(part.get_content())
            else:
                body = msg.get_content()

            return {
                'subject': subject,
                'body': body,
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'date': msg.get('date', ''),
                'full_text': f"{subject} {body}".strip()
            }

        except Exception as e:
            print(f"⚠️ Error parsing email: {e}")
            return {
                'subject': '',
                'body': raw_email,
                'from': '',
                'to': '',
                'date': '',
                'full_text': raw_email
            }

    # -----------------------------------------------------------------
    def process_email(self, raw_email: str, anonymize: bool = True) -> str:
        """
        Full pipeline: parse → clean → anonymize → return processed text.
        """
        parts = self.extract_email_parts(raw_email)
        cleaned = self.clean_text(parts.get('full_text', ''), anonymize=anonymize)
        return cleaned


# ---------------------------------------------------------------------
# ✅ Dataset Loading and Processing
# ---------------------------------------------------------------------
def load_and_preprocess_dataset(
    file_path: str,
    text_column: str = 'text',
    label_column: str = 'label',
    sample_size: int = None,
    anonymize: bool = True,
    min_length: int = 10
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load, preprocess, and clean an email dataset.

    Args:
        file_path: Path to dataset CSV
        text_column: Column containing text
        label_column: Column containing class labels
        sample_size: Optional number of samples to use
        anonymize: Remove PII
        min_length: Minimum text length to keep

    Returns:
        Tuple of (processed DataFrame, list of class names)
    """
    print(f"📂 Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Optional sampling
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"📊 Loaded {len(df)} samples")

    # Initialize preprocessor
    preprocessor = EmailPreprocessor()

    # Use tqdm for progress feedback
    tqdm.pandas(desc="🧹 Cleaning emails")
    df['processed_text'] = df[text_column].progress_apply(
        lambda x: preprocessor.process_email(str(x), anonymize=anonymize)
    )

    # Filter short texts
    df = df[df['processed_text'].str.len() >= min_length].reset_index(drop=True)

    # Collect label info
    classes = sorted(df[label_column].unique().tolist())

    print(f"\n✅ Preprocessing complete!")
    print(f"   - Remaining samples: {len(df)}")
    print(f"   - Classes: {classes}")
    for cls in classes:
        count = (df[label_column] == cls).sum()
        print(f"     - {cls}: {count} ({count / len(df) * 100:.1f}%)")

    return df, classes


# ---------------------------------------------------------------------
# ⚖️ Dataset Balancing
# ---------------------------------------------------------------------
def create_balanced_dataset(
    df: pd.DataFrame,
    label_column: str = 'label',
    samples_per_class: int = 5000
) -> pd.DataFrame:
    """
    Create a balanced dataset by sampling equal numbers per class.
    Uses replacement if some classes are smaller than target.
    """
    print(f"⚖️  Creating balanced dataset ({samples_per_class} per class)...")
    balanced_dfs = []

    for label in df[label_column].unique():
        class_df = df[df[label_column] == label]
        n_samples = min(samples_per_class, len(class_df))
        sampled = class_df.sample(
            n=n_samples,
            random_state=42,
            replace=len(class_df) < samples_per_class
        )
        balanced_dfs.append(sampled)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Balanced dataset created: {len(balanced_df)} total samples")

    return balanced_df


# ---------------------------------------------------------------------
# ✂️ Dataset Splitting
# ---------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification.
    """
    from sklearn.model_selection import train_test_split

    print("\n✂️ Splitting dataset...")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'] if 'label' in df.columns else None,
    )

    val_rel_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        random_state=random_state,
        stratify=train_val_df['label'] if 'label' in train_val_df.columns else None,
    )

    print(f"📊 Dataset split summary:")
    print(f"   - Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------
# 🧪 Example Quick Test (Run standalone)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    preprocessor = EmailPreprocessor()
    sample_email = """
    From: john.doe@example.com
    To: jane.smith@company.com
    Subject: Meeting tomorrow

    Hi Jane,

    Can we meet tomorrow at 3pm? My phone is +1-234-567-8900.
    Check this link: https://example.com/meeting

    Best,
    John
    """

    cleaned = preprocessor.process_email(sample_email)
    print("\nOriginal Email:\n", sample_email)
    print("\n🧹 Cleaned Email:\n", cleaned)
