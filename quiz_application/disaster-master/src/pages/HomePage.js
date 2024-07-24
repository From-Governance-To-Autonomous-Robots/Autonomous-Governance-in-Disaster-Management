import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { db } from '../services/firebaseConfig';
import { useAuth } from '../context/AuthContext';
import '../styles/HomePage.css';
import decisionTreeImage from '../assets/decision_tree.png'; // Import the decision tree image

const HomePage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [feedback, setFeedback] = useState('');

  const handleAgree = async () => {
    if (user) {
      const userDoc = doc(db, 'users', user.uid);
      await updateDoc(userDoc, {
        consent_agree: true
      });
      navigate('/training/victim/checkaid', { state: { task: 'info', phase: 'train' } });
    }
  };

  const handleFeedbackSubmit = async () => {
    const userDoc = doc(db, 'users', user.uid);
    const userDocSnapshot = await getDoc(userDoc);
    const userDocData = userDocSnapshot.data();
    await updateDoc(userDoc, {
      feedback
    });
    alert('Response Recorded');
  };

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen);
  };

  return (
    <div className="home-page">
      <div className="message-box">
        <h1>Welcome to the Disaster Management Survey</h1>
        <div className="section">
          <h2>Project Title</h2>
          <p>Autonomous Governance of Unstructured Decision Making in Disaster Management</p>
        </div>
        <div className="section">
          <h2>Affiliation</h2>
          <p>
            Student: Julian Gerald Dcruz<br/>
            MSc. Applied Artificial Intelligence<br/>
            Cranfield University<br/>
            Supervisors: Dr. Miguel Arana Catania & Dr. Argyrios Zolotas
          </p>
        </div>
        <div className="section">
          <h2>Abstract</h2>
          <p>
            Disaster management often involves unstructured decision-making where human decision-makers are overwhelmed with information from victims, local bodies, and agencies. The complexity of the situation can lead to delayed and suboptimal decisions, affecting relief and rescue efforts.
          </p>
          <p>
            This project proposes a structured governance framework using reinforcement learning to facilitate decision-making in disaster management. By simulating disaster scenarios with annotated datasets, the framework helps to systematically address decision points using AI agents.
          </p>
          <p>
            The framework employs an A2C Reinforcement Learning model, supported by three MultiModal Image+Text classification models and two MultiLabel Image classification models. These AI agents provide confidence estimations at each decision node in a structured decision tree, aiding the RL model in making informed decisions.
          </p>
          <div className="image-container" onClick={toggleModal}>
            <img src={decisionTreeImage} alt="Decision Tree" className="decision-tree-image" />
            <p>Click to view the proposed governance framework depicted as a decision tree.</p>
          </div>
        </div>
        <div className="section">
          <h2>How to participate?</h2>
          <p>
            In this survey, you will make decisions at different levels:
            <ul>
              <li>Informative vs Not informative data</li>
              <li>Type of Informative data</li>
              <li>Type of damage</li>
              <li>Post-disaster damage assessment using satellite data</li>
              <li>Post-disaster damage assessment using drone data</li>
            </ul>
            Your responses will be compared with correct decisions at each level and scored accordingly. Please try to complete at least 5 scenarios, which will require approximately 10 minutes of your time.
          </p>
        </div>
        <div className="feedback-section">
          <h3>Share Your Experience</h3>
          <textarea
            placeholder="Share your experience with decision making, your qualifications, etc."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            className="feedback-textarea"
          />
          <button onClick={handleFeedbackSubmit} className="feedback-submit-button">
            Submit Feedback
          </button>
          <p className="disclaimer-text">This information is collected for socio-cultural characteristic comparisons.</p>
        </div>
        <div className="section">
          <h2>Consent</h2>
          <p>
            By clicking "Agree", you consent to participate in this survey. No personal information will be collected, and all responses will remain anonymous. Your responses will be used for research and publication purposes.
          </p>
        </div>
        <button onClick={handleAgree} className="agree-button">Agree</button>
      </div>
      {isModalOpen && (
        <div className="modal" onClick={toggleModal}>
          <div className="modal-content">
            <img src={decisionTreeImage} alt="Decision Tree" className="modal-image" />
          </div>
        </div>
      )}
    </div>
  );
};

export default HomePage;
