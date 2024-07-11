import React from 'react';
import '../styles/Question.css';

const Question = ({ question, handleAnswer }) => {
  return (
    <div className="question-container">
      <img src={question.image} alt="question" className="question-image" />
      <p className="question-text"><strong>{question.text}</strong></p>
      <p className="question-format"><strong>Question: {question.question_format}</strong></p>
      <div className="button-container">
        {question.question_options.map((option, index) => (
          <button
            key={index}
            onClick={() => handleAnswer(option)}
            className="question-button"
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Question;
