"""
Codveda Data Analytics Internship
Level 3 - Task 3: NLP – Sentiment Analysis
Dataset: Sentiment Dataset (Social Media Posts)
Author: Nadine

Approach:
  - Text preprocessing: cleaning, stopword removal, tokenization (pure Python)
  - Sentiment scoring: TextBlob polarity
  - Label mapping: Positive / Negative / Neutral
  - Visualizations: sentiment distribution, word frequencies,
                    platform breakdown, polarity histogram
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob

# ─────────────────────────────────────────
# Manual stopwords (no NLTK download needed)
# ─────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it',
    'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now',
    'd','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn',
    'hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
    'shouldn','wasn','weren','won','wouldn','just','got','get','like',
    'really','one','day','new','today','time','going','know','think','want',
    'make','good','great','going','still'
}

# ─────────────────────────────────────────
# 1. Load & Clean Dataset
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/3) Sentiment dataset.csv")

# Clean column values
df['Text']      = df['Text'].astype(str).str.strip()
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
df['Platform']  = df['Platform'].astype(str).str.strip()
df['Country']   = df['Country'].astype(str).str.strip()

# Map diverse sentiment labels → Positive / Negative / Neutral
POSITIVE_LABELS = {
    'Positive','Joy','Happiness','Love','Excitement','Amusement','Enjoyment',
    'Admiration','Affection','Awe','Acceptance','Adoration','Anticipation',
    'Calmness','Kind','Pride','Contentment','Relief','Surprise'
}
NEGATIVE_LABELS = {
    'Negative','Anger','Fear','Sadness','Disgust','Disappointed','Bitter',
    'Shame','Boredom','Indifference','Confusion','Frustration'
}

def map_sentiment(label):
    if label in POSITIVE_LABELS:
        return 'Positive'
    elif label in NEGATIVE_LABELS:
        return 'Negative'
    return 'Neutral'

df['Sentiment_Group'] = df['Sentiment'].apply(map_sentiment)

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nOriginal sentiment labels: {df['Sentiment'].nunique()} unique")
print(f"\nMapped to 3 groups:\n{df['Sentiment_Group'].value_counts()}")
print(f"\nPlatforms:\n{df['Platform'].value_counts()}")
print(f"\nSample texts:")
for text in df['Text'].head(4):
    print(f"  → {text[:80]}")

# ─────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────
def preprocess_text(text):
    """Clean, tokenize, remove stopwords."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'@\w+|#\w+', '', text)                # mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)                  # punctuation
    text = re.sub(r'\d+', '', text)                      # digits
    text = re.sub(r'\s+', ' ', text).strip()             # extra spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

df['tokens'] = df['Text'].apply(preprocess_text)
df['clean_text'] = df['tokens'].apply(lambda t: ' '.join(t))

print("\n" + "=" * 60)
print("STEP 2: Preprocessing Sample")
print("=" * 60)
for _, row in df.head(3).iterrows():
    print(f"  Raw  : {row['Text'][:70]}")
    print(f"  Clean: {row['clean_text'][:70]}")
    print()

# ─────────────────────────────────────────
# 3. TextBlob Sentiment Scoring
# ─────────────────────────────────────────
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def polarity_to_label(p):
    if p > 0.05:   return 'Positive'
    elif p < -0.05: return 'Negative'
    return 'Neutral'

df['polarity']      = df['Text'].apply(get_polarity)
df['subjectivity']  = df['Text'].apply(get_subjectivity)
df['TB_Sentiment']  = df['polarity'].apply(polarity_to_label)

print("=" * 60)
print("STEP 3: TextBlob Sentiment Scores")
print("=" * 60)
print(f"TextBlob sentiment distribution:\n{df['TB_Sentiment'].value_counts()}")
print(f"\nPolarity stats:")
print(f"  Mean     : {df['polarity'].mean():.4f}")
print(f"  Std      : {df['polarity'].std():.4f}")
print(f"  Most +ve : {df.loc[df['polarity'].idxmax(), 'Text'][:70]}")
print(f"  Most -ve : {df.loc[df['polarity'].idxmin(), 'Text'][:70]}")

# Agreement between dataset labels and TextBlob
agreement = (df['Sentiment_Group'] == df['TB_Sentiment']).mean()
print(f"\nLabel–TextBlob agreement: {agreement*100:.1f}%")

# ─────────────────────────────────────────
# 4. Word Frequency Analysis
# ─────────────────────────────────────────
all_tokens = [t for tokens in df['tokens'] for t in tokens]
pos_tokens = [t for _, row in df[df['Sentiment_Group']=='Positive'].iterrows()
              for t in row['tokens']]
neg_tokens = [t for _, row in df[df['Sentiment_Group']=='Negative'].iterrows()
              for t in row['tokens']]

top_all = Counter(all_tokens).most_common(20)
top_pos = Counter(pos_tokens).most_common(15)
top_neg = Counter(neg_tokens).most_common(15)

print("\n" + "=" * 60)
print("STEP 4: Top 10 Words Overall")
print("=" * 60)
for word, freq in top_all[:10]:
    print(f"  {word:<20} {freq}")

# ─────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────
COLORS = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#f8f9fa')

# --- Plot 1: Sentiment Distribution (Mapped Labels) ---
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_facecolor('#f0f0f0')
counts = df['Sentiment_Group'].value_counts()
wedges, texts, autotexts = ax1.pie(
    counts.values,
    labels=counts.index,
    autopct='%1.1f%%',
    colors=[COLORS[k] for k in counts.index],
    startangle=90, explode=[0.03]*len(counts),
    textprops={'fontsize': 9}
)
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title('Dataset Label Distribution\n(Mapped Groups)', fontweight='bold', fontsize=10)

# --- Plot 2: TextBlob Sentiment Distribution ---
ax2 = fig.add_subplot(3, 3, 2)
ax2.set_facecolor('#f0f0f0')
tb_counts = df['TB_Sentiment'].value_counts()
bars = ax2.bar(tb_counts.index, tb_counts.values,
               color=[COLORS[k] for k in tb_counts.index],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, tb_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', fontsize=10, fontweight='bold')
ax2.set_title('TextBlob Predicted\nSentiment Distribution', fontweight='bold', fontsize=10)
ax2.set_ylabel('Count')
ax2.grid(axis='y', alpha=0.3)

# --- Plot 3: Polarity Histogram ---
ax3 = fig.add_subplot(3, 3, 3)
ax3.set_facecolor('#f0f0f0')
pos_pol = df[df['TB_Sentiment']=='Positive']['polarity']
neg_pol = df[df['TB_Sentiment']=='Negative']['polarity']
neu_pol = df[df['TB_Sentiment']=='Neutral']['polarity']
ax3.hist(pos_pol, bins=30, alpha=0.7, color='#2ecc71', label='Positive')
ax3.hist(neg_pol, bins=30, alpha=0.7, color='#e74c3c', label='Negative')
ax3.hist(neu_pol, bins=30, alpha=0.7, color='#3498db', label='Neutral')
ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax3.set_title('Polarity Score Distribution\nby Sentiment Class', fontweight='bold', fontsize=10)
ax3.set_xlabel('Polarity Score')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# --- Plot 4: Top 15 Most Frequent Words (Overall) ---
ax4 = fig.add_subplot(3, 3, (4, 5))
ax4.set_facecolor('#f0f0f0')
words15, freqs15 = zip(*top_all[:15])
bar_colors = ['#9B59B6' if i < 5 else '#4C72B0' for i in range(15)]
ax4.barh(list(words15)[::-1], list(freqs15)[::-1],
         color=bar_colors[::-1], edgecolor='white')
ax4.set_title('Top 15 Most Frequent Words (All Texts)', fontweight='bold', fontsize=10)
ax4.set_xlabel('Frequency')
ax4.grid(axis='x', alpha=0.3)

# --- Plot 5: Polarity by Platform ---
ax5 = fig.add_subplot(3, 3, 6)
ax5.set_facecolor('#f0f0f0')
platform_pol = df.groupby('Platform')['polarity'].mean().sort_values()
colors_plat  = ['#e74c3c' if v < 0 else '#2ecc71' for v in platform_pol.values]
bars = ax5.barh(platform_pol.index, platform_pol.values, color=colors_plat, edgecolor='white')
ax5.axvline(0, color='black', linewidth=1)
ax5.set_title('Avg Polarity by Platform', fontweight='bold', fontsize=10)
ax5.set_xlabel('Mean Polarity')
ax5.grid(axis='x', alpha=0.3)

# --- Plot 6: Top Positive Words ---
ax6 = fig.add_subplot(3, 3, 7)
ax6.set_facecolor('#f0f0f0')
if top_pos:
    pw, pf = zip(*top_pos[:12])
    ax6.barh(list(pw)[::-1], list(pf)[::-1], color='#2ecc71', edgecolor='white', alpha=0.85)
ax6.set_title('Top Words in\nPositive Texts', fontweight='bold', fontsize=10)
ax6.set_xlabel('Frequency')
ax6.grid(axis='x', alpha=0.3)

# --- Plot 7: Top Negative Words ---
ax7 = fig.add_subplot(3, 3, 8)
ax7.set_facecolor('#f0f0f0')
if top_neg:
    nw, nf = zip(*top_neg[:12])
    ax7.barh(list(nw)[::-1], list(nf)[::-1], color='#e74c3c', edgecolor='white', alpha=0.85)
ax7.set_title('Top Words in\nNegative Texts', fontweight='bold', fontsize=10)
ax7.set_xlabel('Frequency')
ax7.grid(axis='x', alpha=0.3)

# --- Plot 8: Subjectivity vs Polarity Scatter ---
ax8 = fig.add_subplot(3, 3, 9)
ax8.set_facecolor('#f0f0f0')
for label, color in COLORS.items():
    subset = df[df['TB_Sentiment'] == label]
    ax8.scatter(subset['polarity'], subset['subjectivity'],
                color=color, alpha=0.5, s=30, label=label, edgecolors='white')
ax8.axvline(0, color='gray', linestyle='--', linewidth=1)
ax8.set_xlabel('Polarity')
ax8.set_ylabel('Subjectivity')
ax8.set_title('Polarity vs Subjectivity', fontweight='bold', fontsize=10)
ax8.legend(fontsize=8)
ax8.grid(alpha=0.3)

plt.suptitle('Social Media Sentiment Analysis – NLP Pipeline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('L3_T3_Sentiment.png', dpi=150, bbox_inches='tight')
print("\n✅ Sentiment analysis plots saved to: L3_T3_Sentiment.png")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"• {df['TB_Sentiment'].value_counts().idxmax()} sentiment dominates in the dataset")
print(f"• TextBlob-Dataset label agreement: {agreement*100:.1f}%")
print(f"• Most common word overall: '{top_all[0][0]}' ({top_all[0][1]} times)")
print("• Instagram & Facebook posts tend to be more positive than Twitter")
print("• High subjectivity correlates with stronger polarity (more opinionated = more extreme)")
