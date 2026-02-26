# Push this repo to GitHub

Run these commands **in your terminal** (from this folder).

## 1. Create a new repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `ECON5140-HW1-Problem5` (or any name you like)
3. Leave "Add a README" **unchecked** (you already have one)
4. Click **Create repository**
5. Copy the repo URL, e.g. `https://github.com/YOUR_USERNAME/ECON5140-HW1-Problem5.git`

## 2. Initialize git and push (run in Terminal)

```bash
cd /Users/djsimrikh/ECON5140-HW1-Problem5

# Initialize repo and first commit
git init
git add .
git commit -m "ECON 5140 HW1 Problem 5: Customer Purchase & Time Series"

# Add your GitHub repo (replace with YOUR URL from step 1)
git remote add origin https://github.com/YOUR_USERNAME/ECON5140-HW1-Problem5.git

# Push to GitHub (use main or master)
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` and the repo name with your actual GitHub username and repo URL.

If you use **SSH** instead of HTTPS:

```bash
git remote add origin git@github.com:YOUR_USERNAME/ECON5140-HW1-Problem5.git
```

## One-line script (after creating the repo on GitHub)

After you create the repo on GitHub, you can run:

```bash
cd /Users/djsimrikh/ECON5140-HW1-Problem5 && git init && git add . && git commit -m "ECON 5140 HW1 Problem 5" && git branch -M main && git remote add origin https://github.com/YOUR_USERNAME/ECON5140-HW1-Problem5.git && git push -u origin main
```

(Replace `YOUR_USERNAME/ECON5140-HW1-Problem5` with your repo URL.)
