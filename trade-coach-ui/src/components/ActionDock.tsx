'use client';

import * as React from 'react';

type Props = {
  streaming: boolean;
  followLive: boolean;
  supportVisible: boolean;
  onToggleStreaming(): void;
  onToggleFollowLive(): void;
  onToggleSupport(): void;
};

export default function ActionDock(props: Props) {
  const {
    streaming,
    followLive,
    supportVisible,
    onToggleStreaming,
    onToggleFollowLive,
    onToggleSupport,
  } = props;

  return (
    <>
      {/* Desktop/right rail */}
      <div className="hidden lg:block fixed right-4 top-[88px] z-40">
        <div className="flex flex-col gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--elev-2)]">
          <Toggle label="Follow Live" pressed={followLive} onClick={onToggleFollowLive} />
          <Toggle label="Streaming Data" pressed={streaming} onClick={onToggleStreaming} />
          <Toggle
            label={`${supportVisible ? 'Hide' : 'Show'} Supporting Levels`}
            pressed={supportVisible}
            onClick={onToggleSupport}
          />
        </div>
      </div>

      {/* Mobile bottom bar */}
      <div className="lg:hidden fixed inset-x-0 bottom-0 z-40">
        <div className="mx-auto max-w-7xl">
          <div className="m-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-[var(--elev-2)]">
            <div className="flex items-center justify-between p-2">
              <Toggle small label="Follow Live" pressed={followLive} onClick={onToggleFollowLive} />
              <Toggle small label="Streaming" pressed={streaming} onClick={onToggleStreaming} />
              <Toggle
                small
                label={supportVisible ? 'Hide Levels' : 'Show Levels'}
                pressed={supportVisible}
                onClick={onToggleSupport}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Toggle({
  label,
  pressed,
  onClick,
  small,
}: {
  label: string;
  pressed: boolean;
  onClick(): void;
  small?: boolean;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={`inline-flex items-center justify-center rounded-lg ${
        small ? 'h-9 px-3 text-xs' : 'h-10 px-4 text-sm'
      } bg-[var(--chip)] hover:bg-[var(--border)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]`}
    >
      {label}
    </button>
  );
}
