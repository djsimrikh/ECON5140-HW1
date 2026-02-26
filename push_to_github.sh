#!/bin/bash
# Push ECON5140-HW1-Problem5 to GitHub
# 1. Create a new repo at https://github.com/new (name: ECON5140-HW1-Problem5)
# 2. Set YOUR_GITHUB_URL below, then run: bash push_to_github.sh

YOUR_GITHUB_URL="https://github.com/YOUR_USERNAME/ECON5140-HW1-Problem5.git"
# Or SSH: YOUR_GITHUB_URL="git@github.com:YOUR_USERNAME/ECON5140-HW1-Problem5.git"

set -e
cd "$(dirname "$0")"
git init
git add .
git commit -m "ECON 5140 HW1 Problem 5: Customer Purchase & Time Series"
git branch -M main
git remote add origin "$YOUR_GITHUB_URL"
git push -u origin main
echo "Done! Repo is at: ${YOUR_GITHUB_URL%.git}"
