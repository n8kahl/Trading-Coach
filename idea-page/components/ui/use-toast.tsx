"use client";

import * as React from "react";

import type { ToastProps } from "@/components/ui/toast";

type ToastData = ToastProps & {
  id: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
};

type ToastOptions = Omit<ToastData, "id"> & { id?: string };

const ToastContext = React.createContext<{
  toasts: ToastData[];
  toast: (props: ToastOptions) => string;
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
  const [toasts, setToasts] = React.useState<ToastData[]>([]);

  const toast = React.useCallback((props: ToastOptions) => {
    const { id: providedId, ...rest } = props;
    const id = providedId ?? genId();
    setToasts((current) => [...current, { ...rest, id }]);
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
