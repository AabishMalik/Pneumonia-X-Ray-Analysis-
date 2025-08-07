#! /usr/bin/env bash


echo "setting up the project..."

rye sync
cd frontend
if command -v bun &> /dev/null; then
    bun install
else
    npm install
fi
cd ..


# start the backend server
echo "starting the backend server..."
rye run python -m bwrp.api --host 0.0.0.0 --port 8000 --reload & disown

# start the frontend server
echo "starting the frontend server..."
cd frontend
if command -v bun &> /dev/null; then
    bun run start
else
    npm start
fi
cd ..




