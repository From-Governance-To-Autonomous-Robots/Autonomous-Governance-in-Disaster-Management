import React, { useEffect, useState } from 'react';
import Quiz from '../components/Quiz';
import { useLocation, useNavigate } from 'react-router-dom';
import { checkNavigationHelper } from '../services/navigationHelper';

const DummyQuizPage = () => {
  const location = useLocation();
  const { task, phase } = location.state || {};
  const navigate  = useNavigate()
  useEffect(()=>{
    const { path, state }  = checkNavigationHelper(task, phase);
    navigate( path, { state })
  },[])
  return (
    <div>
    </div>
  );
};

export default DummyQuizPage;
