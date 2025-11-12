# 🔮 InSightGenie — AI-Powered Data Insight Assistant  
*Transforming raw datasets into meaningful stories and actionable insights.*

---

##  What is InSightGenie?  
InSightGenie is an intelligent EDA assistant designed to help everyone — from students to business professionals — explore, visualize, and interpret data without needing to write code.  
Upload any `.csv` or `.xlsx` file and get automatically: data profiling, visualization recommendations, hypothesis testing, and plain-English explanations. It’s analytics made simple, rigorous and accessible.

---

## ✨ Why You Need It  
- **Complex data, simple tools**: Traditional dashboards show “what” is happening. InSightGenie tells you **why**, backing it with statistics.  
- **No coding required**: The hours spent profiling, cleaning, and charting vanish — just upload and go.  
- **Business-ready output**: Beyond charts, get exportable PDF/PPT reports with visual insights and natural-language summaries.

---

## 🎯 Key Features  
- 📋 **Automated Profiling**: Auto-detects data types, missing values, distributions and alerts you.  
- 📈 **Smart Visuals**: Suggests bar charts, scatter plots, heatmaps and more based on your data patterns.  
- 🧮 **Hypothesis Testing**: Runs t-tests, ANOVA, chi-square, and correlation significance with interpretation.  
- 🤖 **AI-Powered Narration**: GPT-driven explanations translate numbers into plain English.  
- 📂 **Exportable Reports**: Download complete insights and visuals as HTML, PDF or PPT for meetings and assignments.

---


### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd AI-Insights-assistant
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here (optional)
   ```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
AI-Insights-assistant/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── feedback.csv                   # User feedback storage
├── README.md                      # This file
│
└── core/                          # Core module directory
    ├── __init__.py
    ├── charts.py                  # Visualization & plotting functions
    ├── eda_plus.py               # Advanced EDA & statistical tests
    ├── exporter.py               # PDF & PowerPoint report generation
    ├── feedback.py               # Feedback collection system
    ├── gemini_helper.py          # Google Gemini API integration
    ├── hypothesis.py             # Hypothesis testing framework
    ├── insights.py               # Automatic insight generation
    ├── loader.py                 # Data loading (CSV, Excel, SQL, PDF)
    ├── ml_quickstart.py          # Quick ML baseline models
    ├── nlq_llm.py               # Natural language query to SQL
    ├── profiler.py              # Data profiling & summarization
    ├── qa.py                    # Q&A engine & Gemini integration
    ├── report.py                # HTML report generation
    ├── safeops.py               # Safe data operations
    ├── suggester.py             # Analysis recommendations
    ├── utils.py                 # Utility functions
    └── fonts/                   # Font resources
```


### Virtual Environment Issues
```bash
# Recreate venv if activation fails
rmdir venv /s /q  # Windows
rm -rf venv        # macOS/Linux
python -m venv venv
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### API Key Errors
- Verify `.env` file exists in project root
- Check API key format is correct
- Ensure keys are valid and have appropriate permissions

### Streamlit Port Issues
```bash
# Run on different port if 8501 is in use
streamlit run app.py --server.port 8502
```

### Data Loading Errors
- Verify CSV/Excel file format is valid
- Check URL is accessible
- Ensure file has headers in first row

---

## 📝 Usage Examples

### Example 1: Upload and Analyze a CSV
1. Click **"Upload CSV or Excel"** in the sidebar
2. Select your file
3. Wait for data to load
4. Explore the **Overview** tab for quick insights
5. Deep dive into **EDA Studio** for detailed analysis

### Example 2: Ask Questions About Your Data
1. Go to **"Ask the Data"** tab
2. Click a suggested question or type your own
3. Choose analysis mode (**Simple** or **Gemini-Powered**)
4. View the visualization and explanation

### Example 3: Generate a Report
1. Complete your analysis using various tabs
2. Go to **Export** tab
3. Click **"Generate PDF Report"** or **"Generate PPT Report"**
4. Download the file

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---


🧑‍💻 Author

Varshita M Doddakallannavar
Master’s Student | Data Analyst | AI & Visualization Enthusiast
varsha.md6562@gmail.com


-----
🔮 Upcoming Features

🗣️ Conversational Query System – Ask questions in natural language.

🔍 Smart Hypothesis Generator – Suggests testable relationships automatically.

📊 Live Data Integration – Connect Google Sheets, SQL, or API sources.

🎙️ Voice-Enabled Analytics – Talk to your dataset in real time.

🤝 Team Collaboration – Share insights with comments and version history.

🧭 Auto-Forecast Module – Predict future trends using time-series AI models.

------

Quote

“Data is not just numbers — it’s a story waiting to be told.
InSightGenie helps you listen.”
