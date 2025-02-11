// Initialize Camera
function startCamera() {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      const video = document.getElementById("video");
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error("Error accessing the camera:", error);
    });
}

// Capture Image and Detect Emotion
document.getElementById("capture-button").addEventListener("click", () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("2d");

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg");

  fetch("/detect_emotion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.emotion) {
        document.getElementById("emotion").textContent = `${data.emotion}`;
        window.location.href = `/recommend?emotion=${encodeURIComponent(
          data.emotion
        )}`;
      } else {
        document.getElementById("emotion").textContent = "No emotion detected";
      }
    })
    .catch((error) => {
      console.error("Error detecting emotion:", error);
    });
});


// Speech Input
document.getElementById("speech-button").addEventListener("click", () => {
  const recognition = new (window.SpeechRecognition ||
    window.webkitSpeechRecognition)();
  const recordButton = document.getElementById("record-button");
  const emotionDisplay = document.getElementById("emotion");
  recognition.lang = "en-US";

  // Prepare the button and UI for recording
  recordButton.disabled = false; // Enable the Record button
  recordButton.textContent = "Recording...";
  emotionDisplay.textContent = "Listening...";

  recognition.onstart = () => {
    console.log("Voice recognition started");
  };

  recognition.onresult = async (event) => {
    const speechInput = event.results[0][0].transcript.toLowerCase();
    console.log("Speech Input:", speechInput);

    emotionDisplay.textContent = "Processing voice...";

    try {
      // Send speech input to the server for analysis
      const response = await fetch("/analyze_pitch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speechText: speechInput }),
      });

      if (!response.ok) throw new Error("Server responded with an error");

      const data = await response.json();

      if (data.emotion) {
        // Update the UI with the detected emotion and redirect
        emotionDisplay.textContent = data.emotion;
        window.location.href = `/recommend?emotion=${encodeURIComponent(
          data.emotion
        )}`;
      } else {
        emotionDisplay.textContent = "Could not determine emotion.";
        alert("Could not determine emotion. Please try again.");
      }
    } catch (error) {
      console.error("Error during emotion analysis:", error);
      emotionDisplay.textContent = "Error occurred. Try again.";
      alert("Error occurred while processing your voice. Please try again.");
    }

    // Reset button state
    recordButton.disabled = true;
    recordButton.textContent = "Start Speaking";
  };

  recognition.onerror = (error) => {
    console.error("Speech recognition error:", error);
    alert("Speech recognition failed. Please try again.");
    // Reset button state
    recordButton.disabled = true;
    recordButton.textContent = "Start Speaking";
    emotionDisplay.textContent = "Error. Try again.";
  };

  recognition.onend = () => {
    console.log("Voice recognition ended");
    // Reset button state
    recordButton.disabled = true;
    recordButton.textContent = "Start Speaking";
    if (!emotionDisplay.textContent.includes("Error")) {
      emotionDisplay.textContent = "Ready for next input.";
    }
  };

  recognition.start();
});

//Text input
document
  .getElementById("text-submit-button")
  .addEventListener("click", async () => {
    const textInput = document.getElementById("text-input").value.trim();
    const emotionDisplay = document.getElementById("emotion");

    // Validate input: only allow letters from Hindi, Kannada, or English
    const regex = /^[a-zA-Z\u0900-\u097F\u0C80-\u0CFF\s]+$/;
    if (!regex.test(textInput)) {
      emotionDisplay.textContent =
        "Invalid input. Please enter only text in English, Hindi, or Kannada.";
      return;
    }

    emotionDisplay.textContent = "Processing text...";

    try {
      // Send input to Flask backend
      const response = await fetch("/analyze_text_tone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });

      // Handle response
      const data = await response.json();
      if (response.ok && data.emotion) {
        emotionDisplay.textContent = data.emotion;
        // Redirect to recommendations if emotion is detected
        window.location.href = `/recommend?emotion=${encodeURIComponent(
          data.emotion
        )}`;
      } else {
        emotionDisplay.textContent =
          data.error || "Could not determine emotion. Please try again.";
      }
    } catch (error) {
      console.error("Error during emotion analysis:", error);
      emotionDisplay.textContent = "An error occurred. Please try again.";
      alert(
        "Error occurred while processing your text. Please check your input and try again."
      );
    }
  });

// Enter from keyboard for text
document.addEventListener("DOMContentLoaded", function () {
  const textInput = document.getElementById("text-input");
  const textSubmitButton = document.getElementById("text-submit-button");

  // Handle the button click event
  textSubmitButton.addEventListener("click", function () {
    submitText();
  });

  // Handle "Enter" key press to trigger the button click
  textInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      textSubmitButton.click();
    }
  });

  function submitText() {
    const text = textInput.value;
    if (text.trim() !== "") {
      console.log("Text submitted: " + text);
      textInput.value = "";
    }
  }
});

// Start camera on load
window.onload = startCamera;
