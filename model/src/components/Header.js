import React from 'react';

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <img src="path_to_your_logo" alt="Mood Mirror Logo" />
      </div>
      <div className="history-link">
        <a href="/history">Our History</a>
      </div>
    </header>
  );
};

export default Header;
