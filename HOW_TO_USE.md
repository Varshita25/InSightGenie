# 🎯 How to Use InSightGenie - Quick Guide

## 📊 Getting Started

### 1. **Upload Your Data**
- **Sidebar**: Click "📤 Upload CSV or Excel" or paste a URL
- Supported formats: CSV, XLSX, XLS
- The app will automatically load and analyze your data

---

## 📑 Tab-by-Tab Guide

### **Overview Tab** 
✅ **Automatically displays** when data is loaded

**What you'll see:**
- Success banner with dataset stats
- 4 metric cards (Rows, Columns, Missing %, Memory)
- Dataset snapshot (first 15 rows)
- Column types and missing values
- Quick visualizations (numeric & categorical distributions)
- Correlation heatmap
- Key insights
- AI-generated summary

**No action needed** - everything loads automatically!

---

### **EDA Studio Tab**
✅ **Automatically displays** detailed analysis

**What you'll see:**
- Dataset info table
- Statistical summary
- Data quality metrics
- Missing value handling preview
- Preprocessing recommendations
- Univariate analysis (select a column)
- Bivariate analysis (select X and Y)
- Multivariate correlation heatmap

**How to use:**
1. Scroll through the automatic analysis
2. Use dropdowns to explore specific columns
3. Try the missing value imputation preview

---

### **Hypotheses Tab** ⚠️ **ACTION REQUIRED**
❌ **Does NOT auto-generate** - you must click the button!

**How to use:**
1. **Select target variable** from dropdown
2. **Adjust significance level (α)** if needed (default: 0.05)
3. **Choose number of tests** (default: 5)
4. **Click "🚀 Run Hypothesis Tests"** button
5. Wait for results to generate

**What you'll see after clicking:**
- Generated hypothesis tests
- H₀ (Null Hypothesis) and H₁ (Alternative Hypothesis)
- Test statistics and p-values
- Decision (Reject/Fail to reject)
- Interpretation
- Visualizations (click to expand)

**Why it doesn't auto-load:**
- Hypothesis testing is computationally intensive
- You need to choose which variable to test against
- Gives you control over the analysis parameters

---

### **✨ Suggested Analyses Tab**
✅ **Automatically displays** recommendations

**What you'll see:**
- Starter questions (natural language)
- Recommended comparisons (auto-generated pairs)
- Rationale for each suggestion
- AI-powered additional ideas

**How to use:**
1. Read through the suggested questions
2. Expand any comparison to see the visualization
3. Each suggestion explains WHY it matters

---

### **💬 Ask the Data Tab** ⚠️ **ACTION REQUIRED**
❌ **Requires your input**

**How to use:**
1. **Click a suggested question** OR type your own
2. **Choose answer mode:**
   - "Simple (local)" - Fast, rule-based answers
   - "Gemini-powered" - AI-powered natural language answers
3. **Wait for the answer**

**What you'll see:**
- Your question displayed
- Visualization (if applicable)
- Technical explanation
- Plain-English insight

---

### **📤 Export Tab**
✅ **Ready to use** anytime

**How to use:**
1. Review what's included in the report
2. Click "📄 Generate PDF Report" OR "📊 Generate PPT Report"
3. Wait for generation (shows spinner)
4. Click "⬇ Download" button when ready

**Report includes:**
- Dataset overview & statistics
- EDA findings
- Data quality assessment
- Hypothesis test results (if you ran them)
- Suggested analyses
- Key visualizations

---

## 🔧 Troubleshooting

### "Hypothesis tab is empty"
**Solution:** You need to click the "🚀 Run Hypothesis Tests" button!

### "No visualizations showing"
**Solution:** 
- Check if you uploaded data
- Try refreshing the page (Ctrl+R)
- Check browser console for errors

### "AI summaries not working"
**Solution:** 
- Check your GOOGLE_API_KEY in .env file
- Verify API key is valid
- Check internet connection

### "App is slow"
**Solution:**
- Large datasets take time to process
- Wait for spinners to complete
- Don't click buttons multiple times

---

## 💡 Pro Tips

### **Best Practices:**
1. **Start with Overview** - Get familiar with your data
2. **Explore EDA Studio** - Understand distributions and relationships
3. **Run Hypotheses** - Test specific relationships
4. **Check Suggestions** - Discover insights you might have missed
5. **Ask Questions** - Use natural language to dig deeper
6. **Export** - Save your findings

### **For Conference Presentations:**
1. Upload your dataset
2. Run hypothesis tests on key variables
3. Take screenshots of:
   - Overview metrics
   - Key visualizations
   - Hypothesis test results
   - AI insights
4. Export PDF/PPT report
5. Use the report as supplementary material

### **Performance Tips:**
- Smaller datasets (\<10,000 rows) work best
- Limit hypothesis tests to 5-10 for faster results
- Use "Simple (local)" mode for Q&A if Gemini is slow

---

## 🎨 UI Features

### **Modern Design Elements:**
- **Gradient headers** - Each tab has a unique color gradient
- **Glassmorphism cards** - Translucent backgrounds with blur
- **Hover effects** - Cards lift and glow on hover
- **Smooth animations** - Professional transitions
- **Color-coded results** - Green for success, amber for warnings

### **Interactive Elements:**
- **Expandable sections** - Click to show/hide content
- **Suggested question buttons** - Click to auto-fill
- **Metric cards** - Hover to see lift effect
- **Download buttons** - Gradient styling with spinners

---

## ❓ Common Questions

**Q: Why do I need to click "Run Hypothesis Tests"?**
A: Hypothesis testing requires you to select a target variable and parameters. It's not automatic because you need to specify what to test.

**Q: Can I test multiple target variables?**
A: Yes! Just change the target variable dropdown and click "Run Hypothesis Tests" again.

**Q: How do I know which test was used?**
A: Each hypothesis result shows the test name (Pearson Correlation, T-Test, ANOVA, Chi-Square).

**Q: What if no hypotheses are generated?**
A: This means your dataset doesn't have enough valid column pairs. Try:
- Uploading a different dataset
- Checking if you have both numeric and categorical columns
- Ensuring you have enough data rows (\>10)

**Q: Can I customize the UI colors?**
A: Yes! See `CUSTOMIZATION_GUIDE.md` for instructions.

---

## 📚 Additional Resources

- **UI_ENHANCEMENTS.md** - Technical documentation of all UI improvements
- **ENHANCEMENT_SUMMARY.md** - Before/after comparison
- **CUSTOMIZATION_GUIDE.md** - How to change colors, fonts, etc.

---

## 🚀 Quick Start Checklist

- [ ] Upload CSV/XLSX file or paste URL
- [ ] Review Overview tab (automatic)
- [ ] Explore EDA Studio tab (automatic)
- [ ] **Click "Run Hypothesis Tests"** in Hypotheses tab
- [ ] Check Suggested Analyses (automatic)
- [ ] Ask questions in Ask the Data tab
- [ ] Export your report (PDF or PPT)

---

**Need Help?** Check the error messages in the app - they provide specific guidance!

**Enjoying the app?** The modern UI is conference-ready! 🎉
