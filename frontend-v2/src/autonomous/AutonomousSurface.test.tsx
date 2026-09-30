import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Launcher } from "./AutonomousSurface";
import { DEFAULT_PROFILE, PROFILES, profileSentence } from "./profiles";

// OD-1 option 4 (accepted): leave the default alone, and make the FIRST launch
// of a new run show the safety-profile chooser with `all` preselected —
// informed consent without changing behaviour. The behaviour was already in the
// tree; nothing pinned it, so a future edit could delete the chooser and no test
// would fail. This is that pin, and it is a test only: no default changes.
//
// `Launcher` is the form the surface draws whenever no run is active, so
// rendering it directly IS "the first launch of a new run". It is not the
// default export, so it is exported by name; the surface fetches on mount, the
// launcher does not, so no fetch needs stubbing.
//
// No `@testing-library/react`: `renderToStaticMarkup` is already a dependency
// (same harness as ModelsSurface.test.tsx).

describe("the first-launch safety-profile chooser", () => {
  const markup = renderToStaticMarkup(<Launcher onLaunched={() => {}} />);

  it("draws the chooser with the default profile preselected", () => {
    expect(markup).toContain('aria-label="Run safety profile"');
    expect(DEFAULT_PROFILE).toBe("all");
    // the controlled <select value="all"> marks the matching option selected
    expect(markup).toMatch(/<option value="all"[^>]*selected/);
    // the chooser offers every profile, not just the preselected one
    for (const p of PROFILES) {
      expect(markup).toContain(`value="${p.value}"`);
    }
  });

  it("renders the selected profile's consequence sentence", () => {
    expect(markup).toContain(profileSentence(DEFAULT_PROFILE));
    expect(markup).toContain("Nothing is withheld");
  });
});
