/**
 * Dump the browser game's own answers for a matrix of bidding, auction and
 * scoring questions, so the Python port can be checked against the original
 * rather than against my reading of it.
 *
 *   node scripts/dump_js_parity.mjs > tests/fixtures/js_parity.json
 *
 * The fixture is committed, so `pytest` never needs node or the other repo.
 * Regenerate it if professional-500's rules change.
 */
import { fileURLToPath } from "node:url";

// The browser game lives beside this repo; override with PROFESSIONAL_500 if it moves.
// Trailing slash matters: without it the last segment is treated as a file name and
// the next relative URL replaces it.
const P500 = process.env.PROFESSIONAL_500
  ? new URL(`file://${process.env.PROFESSIONAL_500.replace(/\/?$/, "/")}`)
  : new URL("../../../professional-500/", import.meta.url);
const src = (name) => fileURLToPath(new URL(`src/${name}`, P500));

const { makeBid, bidValue, bidRank, legalBids, contractLabel, contractShort } =
  await import(src("bid.js"));
const { Auction } = await import(src("auction.js"));
const { scoreContract, matchWinner } = await import(src("scoring.js"));

const ALL_DENOMS = ["S", "C", "D", "H", "NT", "NULLO", "OPEN_NULLO"];

const allBids = [];
for (let tricks = 6; tricks <= 10; tricks++) {
  for (const denom of ["S", "C", "D", "H", "NT"]) allBids.push(makeBid(tricks, denom));
}
allBids.push(makeBid(0, "NULLO"), makeBid(0, "OPEN_NULLO"));

const bids = allBids.map((bid) => ({
  tricks: bid.tricks,
  denom: bid.denom,
  value: bidValue(bid),
  rank: bidRank(bid),
  label: contractLabel(bid),
  short: contractShort(bid),
}));

/* legal bids after a range of previous bids — the whole ladder, not a sample */
const previous = [null, [6, "S"], [6, "NT"], [7, "NT"], [8, "S"], [10, "D"], [0, "NULLO"], [0, "OPEN_NULLO"]];
const legal = previous.map((prev) => {
  const p = prev ? makeBid(prev[0], prev[1]) : null;
  const out = legalBids({ previous: p }).map((b) => [b.tricks, b.denom]);
  return { previous: prev, bids: out };
});

/* scripted auctions: every action in order, then the result */
const transcripts = [
  { first: "south", actions: ["pass", "pass", "pass", "pass"] },
  { first: "south", actions: ["bid:6:S", "pass", "pass", "pass"] },
  {
    first: "west",
    actions: ["bid:6:C", "bid:6:H", "pass", "pass", "bid:7:S", "pass", "pass", "pass"],
  },
  { first: "east", actions: ["bid:6:NT", "pass", "bid:7:NT", "pass", "pass", "pass"] },
  { first: "north", actions: ["pass", "pass", "bid:0:NULLO", "pass", "pass", "pass"] },
  {
    first: "south",
    actions: ["bid:8:S", "pass", "pass", "bid:8:NT", "pass", "pass", "pass"],
  },
];

const auctions = transcripts.map(({ first, actions }) => {
  const auction = new Auction({ first });
  const turns = [];
  for (const action of actions) {
    if (auction.isOver) break;
    turns.push(auction.turn);
    if (action === "pass") auction.pass(auction.turn);
    else {
      const [, tricks, denom] = action.split(":");
      auction.place(auction.turn, makeBid(Number(tricks), denom));
    }
  }
  const result = auction.result;
  return {
    first,
    actions,
    turns,
    finished: auction.isOver,
    result: result && {
      type: result.type,
      seat: result.seat,
      team: result.team,
      contract: result.contract ? [result.contract.tricks, result.contract.denom] : null,
    },
    notes: Object.fromEntries(
      Object.entries(auction.notes()).map(([seat, value]) => [
        seat,
        typeof value === "string" ? value : [value.tricks, value.denom],
      ]),
    ),
  };
});

/* scoring: the full grid over made/failed contracts, slams and both nullos */
const scores = [];
for (const contract of allBids) {
  for (const team of ["us", "them"]) {
    for (let made = 0; made <= 10; made++) {
      const tricksByTeam =
        team === "us" ? { us: made, them: 10 - made } : { us: 10 - made, them: made };
      const r = scoreContract({ contract, contractorTeam: team, tricksByTeam });
      scores.push({
        contract: [contract.tricks, contract.denom],
        contractorTeam: team,
        tricks: tricksByTeam,
        result: {
          made: r.made,
          contractorTricks: r.contractorTricks,
          opponentTricks: r.opponentTricks,
          value: r.value,
          contractorPoints: r.contractorPoints,
          opponentPoints: r.opponentPoints,
          note: r.note,
        },
      });
    }
  }
}

const totalsGrid = [
  { totals: { us: 0, them: 0 } },
  { totals: { us: 499, them: 499 } },
  { totals: { us: 500, them: 499 } },
  { totals: { us: 500, them: 500 }, tieBreak: "us" },
  { totals: { us: 500, them: 500 }, tieBreak: "them" },
  { totals: { us: 480, them: 520 } },
  { totals: { us: 520, them: 480 } },
  { totals: { us: 610, them: 620 } },
];
const matches = totalsGrid.map(({ totals, tieBreak }) => ({
  totals,
  tieBreak: tieBreak ?? "us",
  winner: matchWinner(totals, { target: 500, tieBreak: tieBreak ?? "us" }),
}));

const payload = { bids, legal, auctions, scores, matches };
console.log(JSON.stringify(payload, null, 1));
