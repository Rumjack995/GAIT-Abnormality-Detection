# GitHub Setup Guide

This guide will help you push your gait abnormality detection project to GitHub.

## Prerequisites

✅ Git is installed
✅ Documentation is ready (README, LICENSE, CONTRIBUTING)
✅ .gitignore is configured to exclude large datasets

## Step-by-Step Instructions

### Step 1: Configure Git (First Time Only)

Open PowerShell and run these commands with your information:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 2: Initialize Git Repository

In your project directory, run:

```powershell
git init
```

### Step 3: Create Initial Commit

```powershell
# Add all files (datasets will be automatically excluded by .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Gait abnormality detection system"
```

### Step 4: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in the top-right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `gait-abnormality-detection` (or your preferred name)
   - **Description**: "Deep learning system for automated gait abnormality detection and analysis"
   - **Visibility**: ✅ **Private** (as requested)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### Step 5: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```powershell
# Add GitHub as remote origin (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example:**
```powershell
git remote add origin https://github.com/johndoe/gait-abnormality-detection.git
git branch -M main
git push -u origin main
```

### Step 6: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your code files
3. Verify the repository size is < 100 MB (datasets excluded)
4. Check that README.md displays nicely on the main page

## What Gets Uploaded?

✅ **Included** (uploaded to GitHub):
- Source code (`gait_analysis/`, `web/`, `scripts/`)
- Documentation (`README.md`, `CONTRIBUTING.md`, `LICENSE`)
- Configuration files (`requirements.txt`, `config.yaml`, `.gitignore`)
- Tests (`tests/`)
- Small example files

❌ **Excluded** (NOT uploaded):
- `data/` directory (~31 GB of datasets)
- `venv/` directory (~2 GB virtual environment)
- `web/uploads/` directory (user uploads)
- `__pycache__/` directories
- Model files (`.h5`, `.pkl`)

## Next Steps: Dataset Storage

Since datasets are excluded from GitHub, you need to store them separately:

### Option 1: Google Drive (Recommended)

1. **Upload your `data/` folder to Google Drive**
2. **Make it shareable** (anyone with link can view)
3. **Copy the share link**
4. **Update the download script:**
   - Edit `scripts/download_datasets.py`
   - Replace `YOUR_GOOGLE_DRIVE_ID` with your actual file ID
   - The file ID is in the URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`

### Option 2: OneDrive

1. Upload `data/` folder to OneDrive
2. Create a share link
3. Update `DATASET_SETUP.md` with the download link

### Option 3: Academic/University Storage

If you have access to university cloud storage, use that and provide access instructions.

## Collaborator Workflow

After setting up GitHub and cloud storage, collaborators can:

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/gait-abnormality-detection.git
cd gait-abnormality-detection

# 2. Set up environment
python setup_env.py
venv\Scripts\activate

# 3. Download datasets
python scripts/download_datasets.py

# 4. Verify setup
python scripts/verify_data.py

# 5. Start working!
jupyter lab
```

## Troubleshooting

### "git: command not found"

**Solution**: Restart PowerShell after installing Git, or run:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### "Repository size too large"

**Solution**: Check that `.gitignore` is working:
```powershell
git status
```
If you see `data/` or `venv/` files, they shouldn't be there. Run:
```powershell
git rm -r --cached data/
git rm -r --cached venv/
git commit -m "Remove large directories"
```

### Authentication Issues

GitHub may ask for authentication. Options:

1. **Personal Access Token** (recommended):
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope
   - Use token as password when pushing

2. **GitHub CLI**:
   ```powershell
   winget install --id GitHub.cli
   gh auth login
   ```

## Summary

✅ Git installed and configured
✅ Repository initialized with all code
✅ Large datasets excluded via .gitignore
✅ Comprehensive documentation created
✅ Ready to push to private GitHub repository

**Total repository size**: ~50-100 MB (instead of 30+ GB!)

---

**Need help?** Check [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue on GitHub.
