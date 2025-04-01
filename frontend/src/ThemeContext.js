import React, { createContext, useState, useContext } from 'react';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return React.createElement(
    ThemeContext.Provider,
    { value: { darkMode, toggleDarkMode } },
    children
  );
};

export const useTheme = () => useContext(ThemeContext);