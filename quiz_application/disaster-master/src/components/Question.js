import React, { useState } from 'react';
import '../styles/Question.css';
import HelperPopup from './HelperPopup';

const Question = ({ question, currentTreeLevel, handleAnswer, handleEndGame, mappingDict, task, phase, currentTreeScore, overallScore }) => {
  const [isHelpVisible, setIsHelpVisible] = useState(false);

  const toggleHelp = () => {
    setIsHelpVisible(!isHelpVisible);
  };

  return (
    <div className="question-container">
      <button className="help-button" onClick={toggleHelp}>
        <span className="help-icon">❓</span> Help
      </button>
      {isHelpVisible && <HelperPopup task={task} phase="train" mappingDict={mappingDict} onClose={toggleHelp} />}
      {phase !== "train" && (
        <div className="score-container">
          <h1>Overall Score: {Number(overallScore.toFixed(1))}</h1>
          <h1>Scenario Score: {currentTreeScore}</h1>
        </div>
      )}
      {phase === "train" && (
        <div className="training-text">
          <h1>You are Currently in Training and Will not be Scored for your Decisions.</h1>
          <h1>You have to learn to make correct decisions and then will be Moved to Start Playing Game.</h1>
        </div>
      )}
      <h1 className="scenario-title">Scenario-{phase === "train" ? "Train" : currentTreeLevel}</h1>
      <div className="question-content">
        <div className="image-container">
          <img src={question.image} alt="question" className="question-image" />
        </div>
      </div>
      <p className="question-text"><strong>{question.text}</strong></p>
      <p className="question-format"><strong>Question: {question.question_format}</strong></p>
      <div className="button-container">
        {question.question_options.map((option, index) => (
          <button
            key={index}
            onClick={() => handleAnswer(option)}
            className={`question-button ${option === 'Gather Additional Data' ? 'gather-data-button' : ''}`}
          >
            {option === 'Gather Additional Data' ? 'Unsure. More data needed.' : option}
          </button>
        ))}
      </div>
      {phase !== "train" && (
        <button
          onClick={handleEndGame}
          className="end-game-button"
        >
          End Game
        </button>
      )}
    </div>
  );
};

export default Question;
