# How to Upload to GitHub

## Step-by-Step Guide

### 1. Create a GitHub Repository

1. Go to https://github.com/new
2. **Repository name**: e.g., `curry-retrieval-analysis` or `sc-effectiveness-project`
3. **Description**: "Two-stage text retrieval system + Curry teammate impact analysis"
4. **Visibility**: Choose **Private** (recommended) or **Public**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **"Create repository"**

### 2. Initialize Git Locally

Open terminal in your project directory:

```bash
cd "/Users/anuragraychowdhury/Desktop/side projects/sc effectiveness project"

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Make your first commit
git commit -m "Initial commit: Retrieval system + Curry teammate analysis"
```

### 3. Connect to GitHub and Push

GitHub will show you commands after creating the repo. They'll look like:

```bash
# Add the remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example** (if your username is `anuragraychowdhury` and repo is `curry-retrieval-analysis`):

```bash
git remote add origin https://github.com/anuragraychowdhury/curry-retrieval-analysis.git
git branch -M main
git push -u origin main
```

### 4. Verify

Go to your GitHub repo page - you should see all your files!

---

## Important Notes

### What Gets Uploaded

Your `.gitignore` file excludes:
- ✅ **Data files** (`data/*.csv`, `data/*.json`) - **Good!** Keeps your data private
- ✅ **Python cache** (`__pycache__/`)
- ✅ **API keys** - Your NewsAPI key in `retrieval/run_project.py` will be uploaded

### Security: API Keys

⚠️ **Your NewsAPI key is in `retrieval/run_project.py`**

**Option 1: Remove it before pushing** (recommended)
- Edit `retrieval/run_project.py` and replace the key with a placeholder:
  ```python
  NEWSAPI_KEY = "YOUR_API_KEY_HERE"  # Add your key locally
  ```

**Option 2: Use environment variables** (better practice)
- Create a `.env` file (add to `.gitignore`)
- Load it with `python-dotenv` package
- Never commit API keys

**Option 3: If already pushed**
- GitHub has a way to remove sensitive data, but it's complex
- Consider rotating your API key if it's already public

---

## Quick Commands Summary

```bash
# 1. Navigate to project
cd "/Users/anuragraychowdhury/Desktop/side projects/sc effectiveness project"

# 2. Initialize git
git init
git add .
git commit -m "Initial commit"

# 3. Connect to GitHub (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

---

## Future Updates

After the initial push, to update your repo:

```bash
git add .
git commit -m "Description of changes"
git push
```

That's it! 🚀


