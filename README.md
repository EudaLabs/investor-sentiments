## 🧠 **Bitcoin Tweet Sentiment & Topic Analysis**  

### 📚 **Team Members**  
- **Member:**   [Efe Celik](https://github.com/efeecllk)  
- **Member:**   [Ulas Ucrak](https://github.com/ulasucrak)  
- **Member:**   [Eray Acikgoz](https://github.com/ackgz0)

---

### 🎯 **Project Overview**  
In this study, we aimed to analyze the **sentiment** and **topic distribution** of **Bitcoin-related tweets** using both **synthetic** and **real datasets**. Our goal was to:  
- Predict tweet **sentiments** (*bearish*, *bullish*, *neutral*).  
- Assess the **relationship** between tweet sentiment and **Bitcoin price movements**.  
- Evaluate the **influence of users** based on follower count and verification status.  
- Compare **topic similarity** between synthetic and real datasets.  
- Determine if **influential users** make more **accurate predictions**.  

---

### 📊 **Datasets**  
- **Synthetic Tweet Dataset:** [Hugging Face Dataset](https://huggingface.co/datasets/TimKoornstra/synthetic-financial-tweets-sentiment)  
- **Real Tweet Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)  
- **Bitcoin Price Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/jkraak/bitcoin-price-dataset)  

---

### ⚙️ **Data Preprocessing**  
- **Synthetic Dataset:** Text cleaning, tokenization, stop-word removal, and lemmatization.  
- **Real Dataset:** Applied similar preprocessing steps for consistency.  

---

### 🤖 **Embedding & Model Training**  
- **Word2Vec** was used for text embeddings.  
- **Trained Models:**  
   - Logistic Regression  
   - Neural Network  
   - Long Short-Term Memory (LSTM)  
- **Model Outputs:** Saved for further analysis and prediction tasks.  

---

### 🔄 **Data Merging & Sentiment Prediction**  
- **Merged Data:** Real tweets and Bitcoin price data aligned using timestamps.  
- **Prediction Models:** Logistic Regression and Neural Network were used to label real tweets.  
- **Sentiment Labels:** Bearish, Bullish, Neutral.  

---

### 🧑‍💻 **User Influence Scoring**  
- **Metrics:** Follower count, retweets, verification status.  
- **Accuracy Analysis:** Compared average influence scores for correct and incorrect predictions.  
- **Impact Study:** Analyzed if influential users made more accurate predictions.  

---

### 📈 **Sentiment-Price Analysis**  
- Analyzed how sentiment trends influenced **Bitcoin price changes**.  
- Compared **Logistic Regression** and **Neural Network** results.  

---

### 🛡️ **Trust Analysis**  
- Users were labeled as **'Trust'** or **'Don't Trust'** based on prediction accuracy and influence scores.  

---

### 📝 **Topic Modeling**  
- Applied **LDA (Latent Dirichlet Allocation)** and **DTM (Document-Term Matrix)**.  
- Compared topic distributions between synthetic and real datasets.  

---

### 🛠️ **Sentiment Prediction Application**  
- Developed an **interactive sentiment prediction application** using the **Logistic Regression model**.  
- **Input:** Text (e.g., tweets or comments).  
- **Output:** Sentiment prediction (*bearish*, *bullish*, *neutral*).  

---

### 🏁 **Conclusion**  
This study demonstrated the effectiveness of using **synthetic datasets** for training sentiment analysis models and validated their performance on **real-world data**.  
- Explored the connection between **sentiment trends**, **price changes**, and **user influence**.  
- Despite limitations in measuring **market impact**, the developed application serves as a **practical tool for real-time sentiment analysis**.  

🚀 **Future Work:** Further refine influence measurement techniques and enhance market impact analysis.  

---

🔗 **Explore the Project**  
Stay tuned for updates and feel free to contribute! 🚀  
