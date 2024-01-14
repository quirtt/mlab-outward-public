cd ~/mlab2
eval `ssh-agent -s`
ssh-add ~/.ssh/mlab2_ssh

# If you are on main and there are uncommited changes, switch to a new branch
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" = "main" ] && [[ -z $(git status -s) ]]; then
    git checkout -b $2/$1
fi

# Add everything, commit and push
git add .
git commit -m "end of day commit"
git push --set-upstream origin $branch

# Clean up all working tree changes
git checkout main
git fetch origin
git reset --hard origin/main
git clean -f

# Reset user details
git config --unset user.name
git config --unset user.email
echo "[user]
        name = MLAB Account
        email = 110868426+mlab-account@users.noreply.github.com
" > ~/.gitconfig