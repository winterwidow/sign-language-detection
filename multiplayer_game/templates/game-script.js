const player1Video = document.getElementById('player1-video');
const player2Video = document.getElementById('player2-video');
const startButton = document.getElementById('start-button');
const timerDisplay = document.getElementById('timer');
const wordDisplay = document.getElementById('word-display');
const player1Result = document.getElementById('player1-result');
const player2Result = document.getElementById('player2-result');

const WORDS = ["HELLO", "GOODBYE", "THANKS", "YES", "NO"]; // Replace with your actual gestures
const GAME_DURATION = 30; // seconds

let timerInterval;
let gameActive = false;
let currentWord = '';
let player1Stream, player2Stream;

// Function to start webcam streams for both players
async function startWebcams() {
    try {
        const stream1 = await navigator.mediaDevices.getUserMedia({ video: true });
        player1Video.srcObject = stream1;
        player1Stream = stream1;

        const stream2 = await navigator.mediaDevices.getUserMedia({ video: { video: true } });
        player2Video.srcObject = stream2;
        player2Stream = stream2;
    } catch (err) {
        console.error("Error accessing the webcam: ", err);
        alert("Could not start webcam. Please ensure you have a camera connected and grant permissions.");
    }
}

// Function to start the game
function startGame() {
    gameActive = true;
    startButton.disabled = true;
    startButton.textContent = "Game in Progress...";

    let timeLeft = GAME_DURATION;
    timerDisplay.textContent = timeLeft;

    // Pick a random word from the list
    currentWord = WORDS[Math.floor(Math.random() * WORDS.length)];
    wordDisplay.textContent = currentWord;
    wordDisplay.setAttribute('data-text', currentWord);

    // Reset results
    player1Result.style.opacity = '0';
    player2Result.style.opacity = '0';

    // Start the countdown timer
    timerInterval = setInterval(() => {
        timeLeft--;
        timerDisplay.textContent = timeLeft;
        if (timeLeft <= 0) {
            endGame("Time's Up!");
        }
    }, 1000);

    // Start processing video frames
    processVideoFrames();
}

// Function to end the game
function endGame(message) {
    gameActive = false;
    clearInterval(timerInterval);
    startButton.disabled = false;
    startButton.textContent = "Play Again";

    // Stop webcam tracks
    if (player1Stream) player1Stream.getTracks().forEach(track => track.stop());
    if (player2Stream) player2Stream.getTracks().forEach(track => track.stop());

    wordDisplay.textContent = message;
    wordDisplay.setAttribute('data-text', message);
}

// Function to continuously send video frames to the backend
async function processVideoFrames() {
    if (!gameActive) return;

    // Create a temporary canvas to get a frame from each video feed
    const canvas1 = document.createElement('canvas');
    const context1 = canvas1.getContext('2d');
    canvas1.width = player1Video.videoWidth;
    canvas1.height = player1Video.videoHeight;
    context1.drawImage(player1Video, 0, 0, canvas1.width, canvas1.height);
    const imageData1 = canvas1.toDataURL('image/jpeg');

    const canvas2 = document.createElement('canvas');
    const context2 = canvas2.getContext('2d');
    canvas2.width = player2Video.videoWidth;
    canvas2.height = player2Video.videoHeight;
    context2.drawImage(player2Video, 0, 0, canvas2.width, canvas2.height);
    const imageData2 = canvas2.toDataURL('image/jpeg');

    try {
        // Send frames to the Flask backend for prediction
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player1_frame: imageData1,
                player2_frame: imageData2,
                target_word: currentWord
            })
        });

        const data = await response.json();
        const p1_prediction = data.player1_prediction;
        const p2_prediction = data.player2_prediction;

        // Display the results
        if (p1_prediction) {
            player1Result.textContent = `Predicted: ${p1_prediction}`;
            player1Result.style.opacity = '1';
        } else {
            player1Result.style.opacity = '0';
        }
        
        if (p2_prediction) {
            player2Result.textContent = `Predicted: ${p2_prediction}`;
            player2Result.style.opacity = '1';
        } else {
            player2Result.style.opacity = '0';
        }

        // Check for a winner
        if (p1_prediction === currentWord && p2_prediction === currentWord) {
            endGame("It's a tie!");
        } else if (p1_prediction === currentWord) {
            endGame("Player 1 Wins!");
        } else if (p2_prediction === currentWord) {
            endGame("Player 2 Wins!");
        }

    } catch (error) {
        console.error("Error with prediction API:", error);
    }
    
    // Continue processing frames
    requestAnimationFrame(processVideoFrames);
}

// Event listeners
startButton.addEventListener('click', startGame);

// Start webcams on page load
window.onload = startWebcams;