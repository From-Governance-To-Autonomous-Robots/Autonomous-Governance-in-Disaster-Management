import React, { useState, useEffect } from 'react';
import '../styles/Question.css';

const Question = ({ question,currentTreeLevel, handleAnswer, handleEndGame, mappingDict, task }) => {

  return (
    <div className="question-container">
      <h1 className="scenario-title">Scenario-{currentTreeLevel}</h1>
      <img src={question.image} alt="question" className="question-image" />
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
            {option}
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
