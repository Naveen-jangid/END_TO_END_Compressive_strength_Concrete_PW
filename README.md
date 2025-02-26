# Concrete Strength Prediction using Machine Learning

##  Project Overview
This project predicts the **compressive strength of concrete** using machine learning techniques. The model takes input parameters such as cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age to estimate the concrete's strength in MPa.

##  Project Tasks
1. **Data Preprocessing**: Cleaning and preparing the dataset.
2. **Model Training**: Training a machine learning model (Random Forest Regressor) for prediction.
3. **Model Deployment**: Deploying the model using **Flask** and **Render**.
4. **Web Interface**: Creating a simple front-end to input parameters and get predictions.
5. **GitHub Actions & Workflow**: Setting up CI/CD pipeline for automated deployments.
6. **Model Evaluation**: Comparing different models to determine the best-performing one.

##  Best Performing Model
After training multiple models, **Random Forest Regressor** was found to provide the most accurate predictions. Below is a comparison of model performances:

![Model Performance](./images/model_performance.png)

##  Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/concrete-strength-prediction.git
cd concrete-strength-prediction
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use 'venv\Scripts\activate'
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application Locally
```bash
python app.py
```

Then, open **http://127.0.0.1:5000/** in your browser.

##  Deployment on Render
1. Push your code to GitHub.
2. Sign up on [Render](https://render.com/).
3. Create a **new Web Service** and connect your repository.
4. In the Build Command, use:
   ```bash
   pip install -r requirements.txt
   ```
5. In the Start Command, use:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:10000 app:app
   ```
6. Deploy and get your live URL!

##  GitHub Workflow (CI/CD)
We use GitHub Actions for automated deployment.
- **.github/workflows/deploy.yml** handles testing and deployment.
- Every push to the main branch triggers an automatic deployment.

##  Common Issues & Fixes
**Issue:** `gunicorn: command not found`
- **Solution:** Add `gunicorn` to `requirements.txt` and redeploy.

**Issue:** Internal Server Error
- **Solution:** Check Render logs and ensure all dependencies are installed correctly.

##  Project Screenshots (Optional)
![Input Form](./images/input_form.png)
![Prediction Output](./images/output.png)
![Model Performance](./images/model_performance.png)

## 📜 License
This project is licensed under the MIT License. Feel free to modify and use it!

---
🚀 Happy Coding!

