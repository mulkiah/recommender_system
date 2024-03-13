''' This code for augmenting the current dataset'''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataAugmentation:
    def __init__(self):
        self.customer_interactions_df = pd.read_csv('./data/customer_interactions.csv', sep=",")
        self.purchase_history_df = pd.read_csv('./data/purchase_history.csv', sep=";")
        self.purchase_history_df = self.purchase_history_df.iloc[:, :-4]
        self.product_details_df = pd.read_csv('./data/product_details.csv', sep=";")
        self.product_details_df = self.product_details_df.iloc[:, :-3]
        self.num_entries = 50

    def random_dates(self, start_date, end_date, n=1):
        date_range = (end_date - start_date).days
        random_days = np.random.randint(0, date_range + 1, n).astype(int)
        return [start_date + timedelta(days=int(day)) for day in random_days]

    def augment_customer_interactions(self):
        if self.customer_interactions_df.empty:
            return pd.DataFrame()

        augmented_data = []

        for i in range(self.num_entries):
            new_data = {
                'customer_id': self.customer_interactions_df['customer_id'].max() + i + 1,
                'page_views': np.random.randint(1, 40),
                'time_spent': np.random.randint(1, 6) * 10,
            }

            augmented_data.append(new_data)

        augmented_df = pd.DataFrame(augmented_data)
        return augmented_df

    def augment_purchase_history(self):
        if self.purchase_history_df.empty:
            return pd.DataFrame()

        augmented_data = []

        for i in range(self.num_entries):
            existing_customer_ids = np.arange(1, 30)
            existing_product_ids = np.arange(101, 155)
            new_data = {
                'customer_id': np.random.choice(existing_customer_ids),
                'product_id': np.random.choice(existing_product_ids),
                'purchase_date': (datetime.strptime(self.purchase_history_df['purchase_date'].max(), '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d'),
            }

            augmented_data.append(new_data)

        augmented_df = pd.DataFrame(augmented_data)
        return augmented_df

    def augment_product_details(self):
        if self.product_details_df.empty:
            return pd.DataFrame()

        augmented_data = []

        for i in range(self.num_entries):
            new_data = {
                'product_id': self.product_details_df['product_id'].max() + i + 1,
                'category': np.random.choice(self.product_details_df['category'].dropna().unique()),
                'price': np.random.randint(2, 101) * 10,
                'ratings': round(np.random.uniform(2, 5), 1),
            }

            augmented_data.append(new_data)

        augmented_df = pd.DataFrame(augmented_data)
        return augmented_df

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)
        print(f"Saved {filename}.")

if __name__ == "__main__":
    augmentation = DataAugmentation()
    augmented_interactions = pd.concat([augmentation.customer_interactions_df, augmentation.augment_customer_interactions()], ignore_index=True)
    augmented_history = pd.concat([augmentation.purchase_history_df, augmentation.augment_purchase_history()], ignore_index=True)
    augmented_details = pd.concat([augmentation.product_details_df, augmentation.augment_product_details()], ignore_index=True)

    augmentation.save_to_csv(augmented_interactions, './data/augmented_interactions.csv')
    augmentation.save_to_csv(augmented_history, './data/augmented_history.csv')
    augmentation.save_to_csv(augmented_details, './data/augmented_details.csv')
