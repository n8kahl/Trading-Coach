import '@testing-library/jest-dom/vitest';
import 'whatwg-fetch';
import React from 'react';

(globalThis as unknown as { React: typeof React }).React = React;
