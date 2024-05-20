import React from 'react';
import SearchSelect from './components/SearchSelect';
import './styles/SearchSelect.css';

const columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5'];

function App() {
  return (
    <div className="App">
      <SearchSelect columns={columns} />
    </div>
  );
}

export default App;