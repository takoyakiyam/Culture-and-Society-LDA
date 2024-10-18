import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
                             QWidget)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud


class LDAApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('LDA Topic Modeling - World Important Events')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #f0f0f0;")  # Modern light background

        # Step 1: Layout
        self.layout = QVBoxLayout()

        # Bold title for the dataset
        self.title_label = QLabel("Dataset: 'World Important Events - Ancient to Modern'")
        self.title_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.layout.addWidget(self.title_label)

        # Dataset description
        self.description_label = QLabel(
            "This dataset spans significant historical milestones from ancient times to the modern era, "
            "covering diverse global incidents. It provides a comprehensive timeline of events that "
            "have shaped the world, offering insights into wars, cultural shifts, technological "
            "advancements, and social movements."
        )
        self.description_label.setWordWrap(True)
        self.layout.addWidget(self.description_label)

        # Dropdown for Type of Event
        self.event_type_label = QLabel("Select Type of Event:")
        self.layout.addWidget(self.event_type_label)

        self.event_type_combo = QComboBox()
        self.event_type_combo.addItem("All")  # Option to select all events
        self.layout.addWidget(self.event_type_combo)

        # Load the dataset to populate the combo box
        self.df = pd.read_csv('World Important Dates.csv')
        event_types = self.df['Type of Event'].dropna().unique()
        for event in event_types:
            self.event_type_combo.addItem(event)

        # Input: Number of topics
        self.topic_label = QLabel("Number of Topics:")
        self.layout.addWidget(self.topic_label)
        
        self.topic_count = QSpinBox()
        self.topic_count.setMinimum(1)
        self.topic_count.setValue(5)  
        self.layout.addWidget(self.topic_count)
        
        # Input: Number of Top Words
        self.top_words_label = QLabel("Number of Top Words per Topic:")
        self.layout.addWidget(self.top_words_label)

        self.top_words_count = QSpinBox()
        self.top_words_count.setMinimum(1)
        self.top_words_count.setValue(10)  # Default value
        self.layout.addWidget(self.top_words_count)

        # Button to generate topics
        self.generate_button = QPushButton("Generate Topics")
        self.generate_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.generate_button.clicked.connect(self.run_lda)
        self.layout.addWidget(self.generate_button)

        # Button to generate word clouds
        self.wordcloud_button = QPushButton("Generate Word Clouds")
        self.wordcloud_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.wordcloud_button.clicked.connect(self.generate_wordclouds)
        self.layout.addWidget(self.wordcloud_button)

        # Output: Topic summaries
        self.summary_label = QLabel("Topic Summaries:")
        self.layout.addWidget(self.summary_label)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.layout.addWidget(self.summary_text)

        # Word cloud plot
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Navigation buttons layout
        self.nav_layout = QHBoxLayout()

        # Previous button as an arrow
        self .prev_button = QPushButton("<-")
        self.prev_button.setStyleSheet("background-color: #FFC107; font-weight: bold;")
        self.prev_button.clicked.connect(self.prev_wordcloud)
        self.prev_button.setVisible(False)  # Initially hidden
        self.nav_layout.addWidget(self.prev_button)

        # Next button as an arrow
        self.next_button = QPushButton("->")
        self.next_button.setStyleSheet("background-color: #FFC107; font-weight: bold;")
        self.next_button.clicked.connect(self.next_wordcloud)
        self.next_button.setVisible(False)  # Initially hidden
        self.nav_layout.addWidget(self.next_button)

        self.layout.addLayout(self.nav_layout)
        self.setLayout(self.layout)

        self.impact_text = None
        self.count_vectorizer = None
        self.doc_term_matrix = None
        self.feature_names = None
        self.lda = None
        self.wordclouds = None
        self.current_wordcloud = 0

    def run_lda(self):
        # Filter data based on selected event type
        selected_event = self.event_type_combo.currentText()
        if selected_event != "All":
            filtered_df = self.df[self.df['Type of Event'] == selected_event]
            self.impact_text = filtered_df['Impact'].dropna()
        else:
            self.impact_text = self.df['Impact'].dropna()

        # Apply LDA with the selected number of topics
        n_topics = self.topic_count.value()
        self.count_vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        self.doc_term_matrix = self.count_vectorizer.fit_transform(self.impact_text)
        self.feature_names = self.count_vectorizer.get_feature_names_out()
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda.fit(self.doc_term_matrix)

        # Generate topic summaries and coherence scores
        topic_summaries = self.generate_topic_summaries()
        self.summary_text.setHtml("".join(topic_summaries)) 

        # Reset wordclouds and navigation buttons
        self.wordclouds = None
        self.prev_button.setVisible(False)
        self.next_button.setVisible(False)
        self.current_wordcloud = 0
        self.figure.clear()
        self.canvas.draw()
        
    def generate_topic_summaries(self):
        no_top_words = self.top_words_count.value()  # Get the number from the SpinBox
        topic_summaries = []
        topic_coherence_scores = self.calculate_topic_coherence()

        # HTML style formatting for output
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [self.feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            coherence_score = topic_coherence_scores[topic_idx]

            # Color-coding the coherence score based on value
            if coherence_score > 0.5:
                color = "green"  # High coherence (closer to 1)
            elif coherence_score > 0.3:
                color = "orange"  # Medium coherence
            else:
                color = "red"  # Low coherence

            # Formatting each topic and score with HTML tags
            summary = (f"<b>Topic {topic_idx + 1}:</b> {', '.join(top_words)}<br>"
                    f"<b>Coherence Score:</b> <span style='color:{color};'>{coherence_score:.4f}</span><br><br>")
            topic_summaries.append(summary)

        return topic_summaries

    def calculate_topic_coherence(self):
        """
        Calculate coherence score for each topic using word co-occurrence.
        This uses cosine similarity between word distributions for simplicity.
        """
        word_co_occurrence_matrix = np.dot(self.doc_term_matrix.T, self.doc_term_matrix)  # Co-occurrence matrix
        topic_coherence_scores = []

        for topic in self.lda.components_:
            # Get the indices of the top words
            top_indices = topic.argsort()[:-11:-1]  # Top 10 words
            top_words_matrix = word_co_occurrence_matrix[top_indices, :][:, top_indices]
            
            # Compute pairwise distances between the words in the topic (cosine similarity for example)
            coherence = np.mean(1 - pairwise_distances(top_words_matrix, metric='cosine'))  # Average cosine similarity
            topic_coherence_scores.append(coherence)
        
        return topic_coherence_scores

    def generate_wordclouds(self):
        if self.lda is None:
            self.summary_text.setPlainText("Please generate topics first.")
            return

        if self.wordclouds is not None:
            self.summary_text.setPlainText("Please generate topics again after changing the number of topics.")
            return

        # Generate word clouds for the topics
        no_top_words = 20
        self.wordclouds = []
        for topic_idx, topic in enumerate(self.lda.components_):
            word_freq = {self.feature_names[i]: topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]}
            wc = WordCloud(background_color='#FFFFFF', width=400, height=200).generate_from_frequencies(word_freq)
            self.wordclouds.append(wc)

        self.display_wordcloud(0)

        # Show navigation buttons
        self.prev_button.setVisible(True)
        self.next_button.setVisible(True)

    def display_wordcloud(self, idx):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.wordclouds[idx], interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Topic {idx + 1}", fontsize=14, fontweight='bold')
        self.canvas.draw()

    def prev_wordcloud(self):
        if self.wordclouds is None:
            return
        self.current_wordcloud = (self.current_wordcloud - 1) % len(self.wordclouds)
        self.display_wordcloud(self.current_wordcloud)

    def next_wordcloud(self):
        if self.wordclouds is None:
            return
        self.current_wordcloud = (self.current_wordcloud + 1) % len(self.wordclouds)
        self.display_wordcloud(self.current_wordcloud)

# Run the application
app = QApplication(sys.argv)
window = LDAApp()
window.show()
sys.exit(app.exec_())
