# The rules DurakFish implements

Durak is a folk game with no governing body, so "the rules" is a family
of rules. This document states exactly what the engine does, and names
every point where variants disagree along with the choice made and why.
Nothing here is left to the reader's assumption: if a clause is not in
this file, the engine does not implement it.

**Variant: Classic Podkidnoy Durak (подкидной дурак), two players, 36 cards.**

Explicitly **not** implemented: Perevodnoy (переводной) Durak. There is no
move that passes an attack to the next player by playing a card of the
same rank. The defender's only options are to beat a card or to take —
`DefenseMove` and `TakeCards`, and nothing else exists in `moves.py`, so
the variant cannot leak in by accident.

## 1. Setup

- 36 cards: ranks 6–A in four suits.
- Each player is dealt 6 cards. Dealing is done in blocks (six to player
  0, then six to player 1) rather than alternating. From a shuffled deck
  the two are distributionally identical and the block deal is easier to
  reproduce from a seed.
- The bottom card of the remaining talon is turned face up. Its suit is
  the trump. It stays in the talon and is **the last card drawn**.
- The trump card is public knowledge but is not a location of its own.
  It is counted once, wherever it physically is (talon, later a hand).

**Ambiguity — who attacks first.** Standard: the holder of the lowest
trump. If neither player was dealt a trump (roughly one deal in 170 with
36 cards) tradition calls for a redeal, which would break reproducibility
from a seed. The engine instead falls back to the lowest card overall, by
rank then by canonical suit order. Deterministic, unbiased between seats,
and documented rather than silent. `first_attacker` can also be forced
explicitly, which matters for benchmarking: the natural rule is not seat-
symmetric, so measuring two bots fairly requires alternating the opener.

## 2. Beating a card

`beats(defending, attacking, trump)` is the whole of it:

- same suit → the higher rank wins (this covers trump against trump);
- different suits → only a trump beats a non-trump;
- a non-trump never beats a trump, whatever its rank.

A card never beats itself. Verified exhaustively over all 36 × 36 × 4
combinations against a second implementation written by case analysis.

## 3. The bout

One **bout** (ход) runs as follows.

1. The attacker plays any one card. The opening card of a bout is
   unrestricted.
2. The defender must either beat that card or announce a take.
3. If the defender beat it, the attacker may add another card, or end the
   bout.
4. Added cards must match a rank already on the table.
5. The bout ends when the attacker ends it, and resolves immediately.

**Modelling decision — one card at a time.** In live play an attacker may
slap down several cards at once. The engine requires them one at a time,
with the defender responding to each. The reachable outcomes are
identical — an attacker who wants three cards on the table plays three
in succession — and the sequential form is what makes the game tree well
defined. A consequence worth stating: outside a take, there is never more
than one undefended card, so "which attack must I defend?" never becomes
ambiguous.

**Ambiguity — which ranks may be added.** The engine allows any rank
present anywhere on the table, including on cards played *in defence*.
This is the common Podkidnoy reading. Some house rules count attacking
cards only; `RuleSet.addition_ranks_include_defenses` names the choice,
though the alternative is not implemented and setting it False raises
rather than silently doing the wrong thing.

## 4. How many cards may be thrown in

The limit for a bout is:

```
min(6, number of cards in the defender's hand when the bout began)
```

Both halves matter. The defender's hand size is the rule the spec asked
for; the cap of six is the standard ceiling and binds after a defender
has taken cards and holds more than six. Cards already on the table count
toward the limit — it is a limit on the bout, not on additions.

The limit is fixed when the bout begins and does not change as the
defender plays cards. Its point is that the defender always holds enough
cards to *attempt* a defence of every attack, though of course not
necessarily enough to beat them.

**Not implemented:** the house rule limiting the first bout of a game to
five cards. Its absence is deliberate, not an oversight.

## 5. Taking

If the defender cannot or will not beat the open card, they announce a
take (`TakeCards`).

**The bout does not end there.** This is the rule the game is named for
(*подкидной* = "thrown in"): the attacker may keep adding cards of ranks
on the table, up to the same limit, and the defender picks all of them
up. Cards move only when the attacker ends the bout. Modelling the take
as an immediate end-of-bout would remove one of the game's central
tactical devices — dumping junk on a defender who has already committed
to picking up.

The defender may not resume defending after announcing a take.

## 6. Resolution

When the attacker ends the bout:

**After a complete defence** (*bito*, бито): every card on the table —
attacking and defending alike — is removed from play into the discard
pile. Cards in the discard never return.

**After a take:** every card on the table goes into the defender's hand.

Then hands are refilled, then roles are assigned.

## 7. Refilling

Players draw back up to six cards, **the attacker first, then the
defender**.

Draw order is a real rule and not bookkeeping. When the talon holds fewer
cards than the players need, whoever draws first may take the last card
and so escape being left holding cards at the end. The engine has a test
named for exactly this: drawing first can lose you the game.

A player already holding six or more draws nothing. A defender who has
just taken usually holds more than six and so draws nothing.

## 8. Who attacks next

- **Defence succeeded** → the defender becomes the attacker. The attack
  passes.
- **Defender took** → the *same player attacks again*. The defender has
  effectively forfeited their turn to attack. This is the punishment for
  taking, and getting it backwards would quietly invert the game's whole
  strategic balance.

## 9. Ending the game

A player is **out** when they hold no cards and the talon cannot refill
them. Both conditions are required. Holding zero cards while the talon
still has cards is an ordinary intermediate position, not a win — the
engine tests this explicitly, because it is the easiest endgame bug to
write.

The check happens only at bout resolution, after refilling. So:

- talon empty and exactly one player out → the other player is the
  **durak** (the loser);
- talon empty and both players out in the same resolution → a **draw**.
  Draws are rare but real; under random play they occur in roughly 0.2%
  of games. An engine that cannot represent a draw will eventually
  produce a wrong training label.

A defender who takes cards cannot go out in that bout, by construction.

### Termination

The game cannot run forever, and the reason is worth recording rather
than trusting to a timeout. Once the talon is empty, each bout either:

- ends in a successful defence, moving at least two cards permanently to
  the discard pile; or
- ends in a take, in which case the attacker played at least one card and
  drew nothing, so the attacker's hand strictly shrinks and they remain
  the attacker.

Both quantities are bounded and monotone, so play terminates. While the
talon is non-empty, every bout either shrinks the talon or moves cards to
the discard. The fuzz tests still carry a move ceiling, but it exists to
catch a bug in this argument, not to paper over its absence.

## 10. State machine

Four phases, no more:

| Phase | Who acts | Options |
|---|---|---|
| `ATTACK` | attacker | play a card; end the bout if the table is non-empty |
| `DEFENSE` | defender | beat the open card, or take |
| `TAKING` | attacker | throw in a matching rank, or end the bout |
| `GAME_OVER` | nobody | — |

**Deviation from the specification.** The spec suggested `DEALING`,
`ATTACK`, `DEFENSE`, `ATTACK_ADDITION`, `RESOLUTION`, `GAME_OVER`. Three
of those are dropped:

- `DEALING` is not a decision point; dealing happens inside `new_game`.
- `RESOLUTION` likewise — discarding, taking, refilling and role
  switching all happen atomically inside the `EndAttack` transition.
- `ATTACK_ADDITION` is not distinct from `ATTACK`. The legal moves in
  both are "add a card or end the bout"; whether the table is empty
  already distinguishes the opening card from an addition, and a bout
  cannot be ended before it starts.

The invariant bought by these omissions is worth naming: **every
non-terminal state is one where somebody has a real decision to make.**
Search never steps through bookkeeping nodes, and `get_legal_moves` never
returns an empty list for a live position — which is itself a tested
property.

## 11. Invariants

`GameState.validate()` enforces, and the fuzz tests check on every
position of thousands of games:

1. Every one of the 36 cards is in exactly one place: a hand, the talon,
   the table, or the discard.
2. Hands are in canonical order, so two positions differing only in the
   order cards were picked up hash identically.
3. The trump suit matches the trump card, which is somewhere in play.
4. No defended slot follows an undefended one on the table.
5. Attacks made never exceed the bout limit, which never exceeds six.
6. Phase and table agree (`DEFENSE` has exactly one open attack, last on
   the table; `TAKING` has at least one; `GAME_OVER` has an empty table
   and an empty talon).
7. A durak is only recorded at game over, and holds cards.

## 12. Information model (DurakFish assumption)

The rules above say who may play what. This section says who may *know*
what, which is a separate question and one that live Podkidnoy leaves
unstated. These are DurakFish's assumptions, and they are choices.

**The discard pile is public.** An observer may read the exact cards in
it. `InformationSet.discard` therefore holds real cards, not a count.

This is a deliberate departure from the letter of live play, where the
pile lies face down and some house rules forbid reviewing it. The
justification: card counting is a genuine skill of Durak, and a strong
human player tracks retired cards from memory. Modelling the pile as
unreadable would not make the engine more human — it would just prevent
it from exercising a skill humans do have, and would push every engine
towards reconstructing the pile from the move history anyway. Making it
directly observable is the same information, honestly labelled.

**What remains hidden:**

| Public | Private |
|---|---|
| the discard pile, exactly | other players' hands |
| every card played to the table | the talon's contents |
| the face-up trump card, always | the talon's order |
| the *number* of cards each player holds | which cards a player drew |
| the *number* of cards drawn from the talon | |
| the talon's size | |

**Observation is cumulative.** A card that sat face up on the table and
was then picked up by the defender has been *seen*. It stays observed
even though it is now in a hidden hand. Remembering what you have watched
is not cheating; it is the whole of card counting.

**Observation is not deduction.** Knowing you have seen a card is not the
same as knowing where it is now.
`InformationSet.unobserved_cards()` reports only what has never been
seen, and it is deliberately *not* the set of cards an opponent might
hold — an opponent may also hold cards you watched them take. Closing
that gap requires combining observations with counting, which is the card
tracker's job in Phase 6, not the observer's in Phase 3.

**Consequence for the endgame.** In two players with an empty talon,
nothing is hidden that could not be worked out, and
`is_open_information()` reports that. It reports the *fact*; performing
the deduction is again Phase 6.

## 13. Known limitations

- Two players only. Three- and four-player Podkidnoy adds a second
  attacker and changes both the throw-in rules and the draw order.
  `RuleSet` rejects other player counts loudly rather than pretending.
- Deck sizes other than 36 are structurally supported but untested as
  variants; `hand_size` is not adjusted for them automatically.
- No optional rules: no Perevodnoy transfers, no "first bout limited to
  five", no trump-card exchange with the six of trumps (a widespread
  house rule that this engine does not implement).
