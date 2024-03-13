# TerraStore Recommender System

## Overview
The TerraStore Recommender System is an AI-powered application designed to enhance the marketing strategy of Terra Store, an e-commerce company. The system predicts customer purchase behavior based on historical data and provides insights into which products a customer is likely to purchase next.

## Features
- Predicts the next product a customer is likely to buy.
- Provides personalized recommendations based on customer interactions and purchase history.
- User-friendly web interface for easy interaction.

## Recommender Systems

### Ranking-Based Recommender System
- **Description:** This recommender system ranks products based on their overall ratings and recommends the top-rated products to users.
- **Methodology:** Products are ranked by their average ratings, and the top-ranked products are recommended to users.
- **Implementation:** Implemented using collaborative filtering techniques such as Singular Value Decomposition (SVD)++ and evaluated by RMSE and MAE value provided by surprise libraries.


## Installation
1. Clone the repository:

`git clone https://github.com/mulkiah/recommender_system.git`

`cd recommender_system`


2. Install dependencies:

    `pip install -r requirements.txt`

3. Run the web application:

    `streamlit streamlit run code/app.py`


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for det