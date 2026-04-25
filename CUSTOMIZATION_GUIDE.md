# 🎨 Quick Customization Guide

## How to Customize Your Enhanced UI

### 🎨 Changing Colors

Edit `static/style.css` and modify the `:root` variables:

```css
:root {
    /* Change primary color (purple gradient) */
    --primary-color: #6366F1;  /* Change this hex code */
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    
    /* Change accent color (cyan) */
    --accent-color: #06B6D4;
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    
    /* Change success color (green) */
    --success-color: #10B981;
    
    /* Change backgrounds */
    --bg-primary: #0F172A;      /* Main background */
    --bg-secondary: #1E293B;    /* Secondary background */
    --bg-tertiary: #334155;     /* Tertiary background */
}
```

### 📝 Changing Fonts

In `static/style.css`, update the import and font-family:

```css
/* Change the Google Fonts import */
@import url('https://fonts.googleapis.com/css2?family=YourFont:wght@300;400;600;700&display=swap');

/* Update the font family */
* {
    font-family: 'YourFont', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
```

### 🔲 Adjusting Spacing

Modify spacing variables in `:root`:

```css
:root {
    --spacing-xs: 0.25rem;   /* 4px */
    --spacing-sm: 0.5rem;    /* 8px */
    --spacing-md: 1rem;      /* 16px */
    --spacing-lg: 1.5rem;    /* 24px */
    --spacing-xl: 2rem;      /* 32px */
    --spacing-2xl: 3rem;     /* 48px */
}
```

### 🌓 Light Mode Toggle

To create a light mode, add to `static/style.css`:

```css
/* Light mode variables */
[data-theme="light"] {
    --bg-primary: #F8FAFC;
    --bg-secondary: #F1F5F9;
    --bg-tertiary: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
}
```

### 🎯 Streamlit Theme Settings

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6366F1"        # Primary accent color
backgroundColor = "#0F172A"     # Main background
secondaryBackgroundColor = "#1E293B"  # Widget backgrounds
textColor = "#F1F5F9"          # Text color
font = "sans serif"            # Font family
```

### 🔄 Applying Changes

After making changes:

1. **Save the file**
2. **Refresh your browser** (Ctrl+R or Cmd+R)
3. Streamlit will auto-reload with new styles

---

## 🎨 Color Scheme Presets

### Preset 1: Ocean Blue
```css
--primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
--accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
```

### Preset 2: Sunset Orange
```css
--primary-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
--accent-gradient: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
```

### Preset 3: Forest Green
```css
--primary-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
--accent-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
```

### Preset 4: Royal Purple
```css
--primary-gradient: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
--accent-gradient: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
```

---

## 🛠️ Common Customizations

### Make Buttons Larger
```css
.stButton > button {
    padding: 1rem 2.5rem;  /* Increase padding */
    font-size: 1.1rem;     /* Increase font size */
}
```

### Change Animation Speed
```css
:root {
    --transition-fast: 100ms cubic-bezier(0.4, 0, 0.2, 1);   /* Faster */
    --transition-base: 500ms cubic-bezier(0.4, 0, 0.2, 1);   /* Slower */
}
```

### Adjust Border Radius (Roundness)
```css
:root {
    --radius-md: 0.25rem;   /* Less rounded */
    --radius-lg: 1.5rem;    /* More rounded */
}
```

### Change Glow Intensity
```css
:root {
    --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.6);  /* Stronger glow */
}
```

---

## 📱 Mobile Responsiveness

Adjust mobile breakpoint in `static/style.css`:

```css
@media (max-width: 768px) {
    h1, .main h1 {
        font-size: 2rem !important;  /* Smaller on mobile */
    }
}
```

---

## 🎨 Adding Your Logo

In `app.py`, modify the hero section:

```python
st.markdown("""
    <div class="hero-container">
        <img src="your-logo-url.png" style="width: 100px; margin-bottom: 1rem;">
        <h1 class="hero-title">📊 InSightGenie</h1>
        ...
    </div>
""", unsafe_allow_html=True)
```

---

## 💡 Tips

1. **Test changes incrementally** - Change one thing at a time
2. **Use browser DevTools** - Inspect elements to see applied styles
3. **Keep backups** - Save original files before major changes
4. **Consistency is key** - Stick to your design system
5. **Check contrast** - Ensure text is readable on backgrounds

---

## 🔍 Troubleshooting

### Styles not applying?
- Clear browser cache (Ctrl+Shift+R)
- Check CSS file path in `app.py`
- Verify CSS syntax (missing semicolons, brackets)

### Colors look wrong?
- Check hex codes are valid (#RRGGBB format)
- Ensure rgba values are 0-1 for alpha
- Verify gradient syntax

### Animations not smooth?
- Check transition timing
- Ensure transform properties are correct
- Test in different browsers

---

**Happy Customizing! 🎨**
