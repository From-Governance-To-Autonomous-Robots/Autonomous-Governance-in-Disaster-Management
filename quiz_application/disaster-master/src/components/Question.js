import React from 'react';
import '../styles/Question.css';

const Question = ({ question, handleAnswer, mappingDict, task }) => {
  return (
    <div className="question-container">
      <img src={question.image} alt="question" className="question-image" />
      <p className="question-text"><strong>{question.text}</strong></p>
      <p className="question-format"><strong>Question: {question.question_format}</strong></p>
      <p><strong>Correct Answer: {mappingDict[task][question.correct_answer]}</strong></p>
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
