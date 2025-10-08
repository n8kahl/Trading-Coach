import React from 'react'

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'openai-chatkit': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        'client-secret'?: string
        'workflow-id'?: string
        'user-id'?: string
      }
    }
  }
}

