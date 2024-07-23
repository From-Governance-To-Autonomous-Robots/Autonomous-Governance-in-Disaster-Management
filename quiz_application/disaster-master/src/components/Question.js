import React from 'react';
import '../styles/Question.css';

const Question = ({ question, currentTreeLevel, handleAnswer, handleEndGame, mappingDict, task, currentTreeScore, overallScore }) => {
  return (
    <div className="question-container">
      <div className="score-container">
        <h1>Overall Score: {Math.round(overallScore * 100)}</h1>
        <h1>Scenario Score: {Math.round(currentTreeScore * 100)}</h1>
      </div>
      <h1 className="scenario-title">Scenario-{currentTreeLevel}</h1>
      <div className="image-container">
        <img src={question.image} alt="question" className="question-image" />
      </div>
      <p className="question-text"><strong>{question.text}</strong></p>
      <p className="question-format"><strong>Question: {question.question_format}</strong></p>
      <p><strong>Correct Answer: {mappingDict[task][question.correct_answer]}</strong></p>
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
      <button
        onClick={handleEndGame}
        className="end-game-button"
      >
        End Game
      </button>
    </div>
  );
};

export default Question;
