"use client";

import * as React from "react";

import type { ToastProps } from "@/components/ui/toast";

type Toast = ToastProps & { id: string };

const ToastContext = React.createContext<{
  toasts: Toast[];
  toast: (props: ToastProps) => string;
  dismiss: (toastId?: string) => void;
}>({
  toasts: [],
  toast: () => "",
  dismiss: () => undefined,
});

export const useToast = () => React.useContext(ToastContext);

let count = 0;
function genId() {
  count += 1;
  return `toast-${count}`;
}

export function ToastProviderCustom({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([]);

  const toast = React.useCallback((props: ToastProps) => {
    const id = props.id ?? genId();
    setToasts((current) => [...current, { ...props, id }]);
    return id;
  }, []);

  const dismiss = React.useCallback((toastId?: string) => {
    if (!toastId) {
      setToasts([]);
      return;
    }
    setToasts((current) => current.filter((toast) => toast.id !== toastId));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, toast, dismiss }}>
      {children}
    </ToastContext.Provider>
  );
}
