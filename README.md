# ✈️ Flight Ticket Price Prediction  

This project implements a machine learning-powered flight ticket price predictor.  
The application is built with **Gradio**, providing an interactive user interface where users can input flight details and get an estimated ticket price.  

---

# ✨ Features  

- Predicts flight ticket price based on historical data.  
- User-friendly interface built with **Gradio**.  
- Takes into account various parameters:  
  - Airline  
  - Source City  
  - Destination City  
  - Flight Duration (hours & minutes)  
  - Total Stops  
  - Date of Journey (day, month, year)  
- Prevents invalid inputs (e.g., same source and destination, past dates).  
<img width="1669" height="819" alt="image" src="https://github.com/user-attachments/assets/54fd46fe-5af1-4e34-907d-ac7a17f22217" />


---

# 📂 Project Structure  

- `app.py` : The main Gradio application file.  
- `flights.csv` : The dataset used for training the model.  
- `flight_price_predictor.pkl` : The trained Random Forest model (auto-saved if not found).  
- `requirements.txt` : Lists all Python dependencies required to run the application.  
- `README.md` : Project documentation.  
- `.gitattributes` : Git settings file.  

---

# 🧠 Model Training  

- The model is trained using a **Random Forest Regressor** on historical flight data.  
- Preprocessing includes:  
  - Filling missing values with mode.  
  - Extracting Day, Month, and Year from the Date of Journey.  
  - Converting flight duration into minutes.  
  - Label encoding categorical features (Airline, Source, Destination, Stops).  
- The model auto-trains if no pre-trained model is found, then saves it as `flight_price_predictor.pkl`.  

---

# 📦 Dependencies  

The project relies on the following Python libraries:  

- gradio  
- pandas  
- numpy  
- scikit-learn  

---

# 🤝 Contributing  

Contributions are welcome! Please feel free to open issues or submit pull requests.  

---

# **📬 Contact**  

💡 For questions, suggestions, or collaborations, reach out:  

- GitHub: [https://github.com/SWAROOP323](https://github.com/SWAROOP323)  
- Email: **swaroopmanchala323@gmail.com**  
