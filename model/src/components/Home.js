import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import feedbackImage from './img.svg';
import Camera from './Camera';

const Home = () => {
  const [emotion, setEmotion] = useState('');
  const [isAnalysisPaused, setIsAnalysisPaused] = useState(false);
  const [isAnalysisStopped, setIsAnalysisStopped] = useState(false);
  const progressBarRef = useRef(null);
  const videoRef = useRef(null);

  const handleCameraStream = (stream) => {
    videoRef.current.srcObject = stream;
  };

  const pauseAnalysis = () => {
    setIsAnalysisPaused(true);
  };

  const stopAnalysis = () => {
    setIsAnalysisStopped(true);
  };

  useEffect(() => {
    const captureFrame = () => {
      if (!isAnalysisPaused && !isAnalysisStopped && videoRef.current) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const { videoWidth, videoHeight } = videoRef.current;
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
        const imageData = canvas.toDataURL('image/jpeg');
        detectEmotion(imageData);
      }
    };

    const intervalId = setInterval(captureFrame, 1000); // Capture frame every second

    return () => clearInterval(intervalId);
  }, [isAnalysisPaused, isAnalysisStopped]);

  const detectEmotion = async (imageData) => {
    try {
      const response = await axios.post('http://localhost:5000/analyze_emotion', { image_data: imageData });
      const { emotion } = response.data;
      setEmotion(emotion);
    } catch (error) {
      console.error('Error detecting emotion: ', error);
    }
  };

  return (
    <div className="home">
      <h2>Welcome to Mood Mirror</h2>
      <p>Real-time emotion detection and feedback generation for products.</p>
      <div className="content">
        <div className="left">
          <h3>Let's see what they are thinking...</h3>
          <div id="camera-container" className="camera-container">
            <Camera onStream={handleCameraStream} />
            <video ref={videoRef} style={{ display: 'none' }} />
          </div>
          <button onClick={pauseAnalysis}>Pause Analysis</button>
          <button onClick={stopAnalysis}>Stop Analysis</button>
          <br />
          <progress ref={progressBarRef} value="0" max="100"></progress>
          <br />
          <p>Detected Emotion: {emotion}</p>
        </div>
        <div className="right">
          <img src={feedbackImage} alt="Feedback" />
        </div>
      </div>
    </div>
  );
};

export default Home;
