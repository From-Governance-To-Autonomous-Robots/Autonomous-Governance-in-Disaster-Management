import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import '../styles/Score.css';
import volcanoGif from '../assets/volcano.gif'; // Import the local GIF file

const Score = () => {
  const [correctlyAnswered, setCorrectlyAnswered] = useState(null);
  const [wronglyAnswered, setWronglyAnswered] = useState(null);
  const [additionalDataRequested, setAdditionalDataRequested] = useState(null);
  const [feedback, setFeedback] = useState('');
  const { user } = useAuth();

  const fetchScoreData = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();

    setCorrectlyAnswered(Math.round(userDocData.CorrectlyAnswered * 100));
    setWronglyAnswered(Math.round(userDocData.WronglyAnswered * 100));
    setAdditionalDataRequested(Math.round(userDocData.GatherAdditionalDataRequested * 100));
  };

  const handleFeedbackSubmit = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    await updateDoc(userDoc, {
      feedback
    });
    alert('Feedback submitted successfully!');
  };

  useEffect(() => {
    fetchScoreData();
  }, [user.uid]);

  return (
    <div className="score-container1">
      <div className="score-display1">
        <h2>Congratulations!</h2>
        <div className="completion-message">
          <h3>You have completed the game!</h3>
          <div className="volcano-container">
            <img src={volcanoGif} alt="Volcanic Eruption" className="volcano-gif" />
          </div>
        </div>
        <div className="score-section">
          <h3>Correctly Answered</h3>
          <div className="score-bar">
            <div className="score-bar-inner correct" style={{ width: `${correctlyAnswered}%` }}>
              {correctlyAnswered}%
            </div>
          </div>
        </div>
        <div className="score-section">
          <h3>Wrongly Answered</h3>
          <div className="score-bar">
            <div className="score-bar-inner wrong" style={{ width: `${wronglyAnswered}%` }}>
              {wronglyAnswered}%
            </div>
          </div>
        </div>
        <div className="score-section">
          <h3>Additional Data Requested</h3>
          <div className="score-bar">
            <div className="score-bar-inner additional" style={{ width: `${additionalDataRequested}%` }}>
              {additionalDataRequested}%
            </div>
          </div>
        </div>
        <div className="feedback-section">
          <h3>Share Your Experience</h3>
          <textarea
            placeholder="Share your experience with how you felt about making the decisions in the game, etc."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            className="feedback-textarea"
          />
          <button onClick={handleFeedbackSubmit} className="feedback-submit-button">
            Submit Feedback
          </button>
          <p className="disclaimer-text">This information is collected for socio-cultural characteristic comparisons.</p>
        </div>
      </div>
    </div>
  );
};

export default Score;
