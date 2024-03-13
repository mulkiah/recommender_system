import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import cross_validate
from datetime import datetime

class CustomerPurchaseRecommendation:
    def __init__(self):
        self.customer_interactions_df = pd.read_csv('./data/augmented_interactions.csv', sep=",")
        self.product_details_df = pd.read_csv('./data/augmented_details.csv', sep=",")
        self.purchase_history_df = pd.read_csv('./data/augmented_history.csv', sep=",")
        self.merged_data_filename = './data/merged_data.csv'

    def merge_data(self):
        merged_dataset = pd.merge(self.customer_interactions_df, self.purchase_history_df, on='customer_id', how='inner') 
        customer_purchase_history = pd.merge(self.product_details_df, merged_dataset, on='product_id', how='inner') 
        customer_purchase_history.to_csv(self.merged_data_filename, index=False)
        return pd.read_csv(self.merged_data_filename)

    def create_features(self, customer_purchase_history):
        '''Create additional features such as purchase sum, purchase count, purchase average and recency'''
        customer_purchase_sum = customer_purchase_history.groupby('customer_id')['price'].sum().reset_index(name='total_purchase')
        customer_purchase_count = customer_purchase_history.groupby('customer_id')['product_id'].count().reset_index(name='num_purchases')
        customer_average_purchase = customer_purchase_history.groupby('customer_id')['price'].mean().reset_index(name='avg_purchase')
        customer_last_purchase_date = customer_purchase_history.groupby('customer_id')['purchase_date'].max().reset_index(name='last_purchase_date')

        customer_purchase_history = pd.merge(customer_purchase_history, customer_purchase_sum, on='customer_id', how='left')
        customer_purchase_history = pd.merge(customer_purchase_history, customer_purchase_count, on='customer_id', how='left')
        customer_purchase_history = pd.merge(customer_purchase_history, customer_average_purchase, on='customer_id', how='left')
        customer_purchase_history = pd.merge(customer_purchase_history, customer_last_purchase_date, on='customer_id', how='left')
        return customer_purchase_history

    def engagement_score(self, customer_purchase_history):
        '''Create additional features: engagement score based on customer activity'''
        customer_purchase_history['normalized_page_views'] = customer_purchase_history['page_views'] / customer_purchase_history['page_views'].max()
        customer_purchase_history['normalized_time_spent'] = customer_purchase_history['time_spent'] / customer_purchase_history['time_spent'].max()

        weight_page_views = 0.5
        weight_time_spent = 0.5

        customer_purchase_history['engagement_score'] = (weight_page_views * customer_purchase_history['normalized_page_views'] +
                                                         weight_time_spent * customer_purchase_history['normalized_time_spent'])

        customer_purchase_history = customer_purchase_history.drop(['normalized_page_views', 'normalized_time_spent'], axis=1)
        return customer_purchase_history

    def load_surprise_dataset(self, customer_purchase_history):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(customer_purchase_history[['customer_id', 'product_id', 'ratings']], reader)
        return data

    def train_recommendation_model(self, data):
        algo = SVDpp()
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        return algo

    def run_streamlit_app(self):
        st.title("Customer Purchase History Recommender System")
        user_id_input = st.number_input("Enter Customer ID:", min_value=1, max_value=self.customer_purchase_history['customer_id'].max(), value=1)
        user_history = self.customer_purchase_history[self.customer_purchase_history['customer_id'] == user_id_input]
        st.subheader("User's Purchase History:")
        st.write(user_history[['product_id', 'ratings']])

        recommendations_ratings = [(product_id, self.algo.predict(user_id_input, product_id).est)
                                   for product_id in self.customer_purchase_history['product_id'].unique()]

        sorted_recommendations_ratings = sorted(recommendations_ratings, key=lambda x: x[1], reverse=True)

        st.subheader("Top Product Recommendations:")
        for i, (product_id, est_rating) in enumerate(sorted_recommendations_ratings[:5]):
            st.write(f"{i + 1}. Product ID: {product_id}")

    def run(self):
        self.customer_purchase_history = self.merge_data()
        self.customer_purchase_history = self.create_features(self.customer_purchase_history)
        self.customer_purchase_history = self.engagement_score(self.customer_purchase_history)
        data = self.load_surprise_dataset(self.customer_purchase_history)
        self.algo = self.train_recommendation_model(data)
        self.run_streamlit_app()

if __name__ == "__main__":
    app = CustomerPurchaseRecommendation()
    app.run()
