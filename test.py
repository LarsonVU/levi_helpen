# import libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import math
from io import BytesIO


import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

def string_to_list(s):
    return [str(x) for x in s.strip('[]').split(',')]


def prepare_train_df(basis_df, category_column, text_column):
    train_df = basis_df.copy()
    train_df = train_df.dropna(subset=[category_column, text_column])
    train_df[category_column] = train_df[category_column].apply(string_to_list)
    return train_df

def create_category_dict(df, sentence_column, category_column):
    category_dict = {}
    
    # Iterate over the DataFrame rows
    for _, row in df.iterrows():
        sentence = row[sentence_column]
        categories = row[category_column]
        
        # Add the sentence to each category in the dictionary
        for category in categories:
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(sentence)
    
    return category_dict

def create_model(train_df, category_column, sentence_column):
    # Load the multilingual sentence transformer
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # Predefined categories with example phrases (you can expand these later!)
    category_examples = create_category_dict(train_df, sentence_column , category_column)

    # Encode all category examples in advance
    category_embeddings = {
        category: model.encode(examples, convert_to_tensor=True)
        for category, examples in category_examples.items()
    }

    return model, category_embeddings

def match_semantic_categories(model, category_embeddings, text, threshold=0.6):
    # Encode input sentence
    if not isinstance(text, str):
        return [None]

    input_embedding = model.encode(text, convert_to_tensor=True)
    
    matched = []
    for category, embeddings in category_embeddings.items():
        # Compute cosine similarities
        cosine_scores = util.cos_sim(input_embedding, embeddings)
        max_score = float(cosine_scores.max())  # Take the highest match
        
        if max_score >= threshold:
            matched.append((category, max_score))
    
    if matched:
        matched.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in matched]
    else:
        return ["(Nieuwe categorie nodig)"]


def assign_labels(model, category_embeddings, df, text_column, category_column):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Apply the matching function to each row in the DataFrame
    df_copy[category_column + "_ML"] = df_copy[text_column].apply(lambda x: match_semantic_categories(model, category_embeddings, x))
    return df_copy


def analyze_dutch_sentiment(dutch_sentence):
    if not isinstance(dutch_sentence, str):
        return [None]

    if not re.search(r'[a-zA-Z]', dutch_sentence):
        return {
            'original': dutch_sentence,
            'translated': None,
            'sentiment': 'Invalid - No words',
            'scores': None
        }
    # Step 1: Translate Dutch to English
    english_sentence = GoogleTranslator(source='nl', target='en').translate(dutch_sentence)

    # Step 2: Analyze sentiment with VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(english_sentence)

    # Step 3: Classify based on compound score
    compound = sentiment_scores['compound']
    if compound >= 0.1:
        sentiment = 'Positive'
    elif compound <= 0.1:
        sentiment = 'Negative'

    return sentiment #{
    #     'original': dutch_sentence,
    #     'translated': english_sentence,
    #     'sentiment': sentiment,
    #     'scores': sentiment_scores
    # }

def assign_sentiment(df, text_column):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Apply the matching function to each row in the DataFrame
    df_copy[text_column + "_sentiment"] = df_copy[text_column].apply(lambda x: analyze_dutch_sentiment(x))
    return df_copy

def main(): 
    # Streamlit app
    st.title("Semantic Category Assignment")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Create Model", "Assign Labels", "Assign Sentiments"])

    basis_df = pd.read_excel("digitale wereld submap.xlsx")
    basis_df = basis_df.dropna(subset = ['@digitaal_watgaatergoed_categorie', '@digitaal_watgaatergoedd', "@digitaal_watkanerbeterr" , "@digitaal_watkanerbeter_categorie"])

    # Tab 1: Train model
    with tab1:
        st.header("Train New Model")
        st.write("Choose to train a model using the basic data or upload your own data.")

        # Radio button to select training option
        training_option = st.radio("Select training option:", ("Use basic data", "Upload your own data"))

        if training_option == "Use basic data":
            # Input for column names
            column_options = basis_df.columns.tolist()
            train_text_column = st.selectbox("Select the text column for training:", column_options)
            train_category_column = st.selectbox("Select the category column for training:", column_options)
            
        elif training_option == "Upload your own data":
            train_file = st.file_uploader("Choose an Excel file for training", type=["xlsx"], key="train_file")

            if train_file is not None:
                # Load the uploaded training Excel file
                train_df = pd.read_excel(train_file)
                st.write("Preview of the training file:")
                st.dataframe(train_df.head())

                # Input for column names
                column_options = train_df.columns.tolist()
                train_text_column = st.selectbox("Select the text column for training:", column_options)
                train_category_column = st.selectbox("Select the category column for training:", column_options)

        train_model = st.button("Train Model")
        if train_model:
            try:
                if train_text_column and train_category_column:
                        # Prepare the training DataFrame
                        prepared_train_df = prepare_train_df(basis_df, train_category_column, train_text_column)


                        # Create the model and category embeddings
                        st.session_state["model"], st.session_state["category_embeddings"]  = create_model(prepared_train_df, train_category_column, train_text_column)

                        st.success("New model and embeddings have been created successfully using your uploaded data!")
            except Exception as e:
                st.error(f"An error occurred while training with your data: {e}")

    # Tab 2: Run algorithm
    with tab2:
        st.header("Category Assignment")
        uploaded_file = st.file_uploader("Choose an Excel file for labeling", type=["xlsx"])
        
        if uploaded_file is not None:
            # Load the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            st.write("Preview of the uploaded file:")
            st.dataframe(df.head())

            # Input for column names
            column_options = df.columns.tolist()
            text_column = st.selectbox("Select the text column for testing:", column_options)
            category_column = st.selectbox("Select the category column for testing:", column_options)

            assign_label = st.button("Assign Labels", key = "assign_label")

            model = st.session_state.get("model", None)
            category_embeddings = st.session_state.get("category_embeddings", None)

            if assign_label and model is not None:
                try:
                    # Assign labels
                    labeled_df = assign_labels(model, category_embeddings, df, text_column, category_column)

                    # Display the labeled DataFrame
                    st.write("Labeled DataFrame:")
                    st.dataframe(labeled_df)

                    # Download the labeled DataFrame

                    buffer = BytesIO()
                    labeled_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Labeled Excel",
                        data=buffer,
                        file_name="labeled_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            elif assign_label and model is None:
                st.warning("Please train the model first before assigning labels.")

    with tab3:
        st.header("Sentiment Analysis")
        uploaded_file = st.file_uploader("Choose an Excel file for sentiment analysis", type=["xlsx"])
        
        if uploaded_file is not None:
            # Load the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            st.write("Preview of the uploaded file:")
            st.dataframe(df.head())

            # Input for column names
            column_options = df.columns.tolist()
            text_column_sent = st.selectbox("Select the text column for sentiment analysis:", column_options)

            determine_sentiment = st.button("determine_sentiment", key = "assign_sent")

            if determine_sentiment:
                try:
                    # Assign labels
                    labeled_df = assign_sentiment(df, text_column_sent)

                    # Display the labeled DataFrame
                    st.write("Labeled DataFrame:")
                    st.dataframe(labeled_df)

                    # Download the labeled DataFrame

                    buffer = BytesIO()
                    labeled_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Sentiment Excel",
                        data=buffer,
                        file_name="Sent_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()