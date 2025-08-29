import React from "react";
import { Link, useNavigate } from "react-router-dom";

type Props = {
  title: string;
  right?: React.ReactNode;
  divider?: boolean;
  backTo?: string;          // if given, go to this path
  onBack?: () => void;      // optional override
};

export default function Backbar({ title, right, divider = true, backTo, onBack }: Props) {
  const navigate = useNavigate();

  const Left = backTo ? (
    <Link
      to={backTo}
      className="flex items-center gap-2 text-gray-700 hover:text-gray-900"
      aria-label="Back"
    >
      <span className="text-2xl leading-none">←</span>
      <span className="text-lg"></span>
    </Link>
  ) : (
    <button
      onClick={() => (onBack ? onBack() : navigate(-1))}
      className="flex items-center gap-2 text-gray-700 hover:text-gray-900"
      aria-label="Back"
    >
      <span className="text-2xl leading-none">←</span>
      <span className="text-lg">Back</span>
    </button>
  );

  return (
    <div className={`flex items-center justify-between py-4 ${divider ? "border-b" : ""}`}>
      <div className="flex items-center gap-4">
        {Left}
        <h1 className="text-2xl font-semibold">{title}</h1>
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
  );
}
