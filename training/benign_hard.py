"""Synthetic benign-hard emails: instruction-like but benign, plus long repetitive benign emails.

Every record is labeled benign (0), carries synthetic=True, its family and template id, and the generator
version. All names, companies and addresses are fictional (example.com / example.org domains).

Families: it_notice, forwarding_request, policy_reminder, calendar_ask (short, imperative, addressed to
a human recipient) and newsletter_long, log_digest_long (long, repetitive). Each family has 4 templates.
Template 0 of every family goes to val (dev); templates 1-3 go to the frozen test, so the frozen slice
is built from templates Stage 1 never had its threshold tuned on.

Deterministic: python -m training.benign_hard prints a sample; build_splits imports generate().
"""
from __future__ import annotations

import random

GENERATOR_VERSION = "benign-hard-v1"
DEV_TEMPLATE = 0  # template id assigned to val; others -> frozen_test

FIRST = ["Alex", "Priya", "Jordan", "Mei", "Samuel", "Fatima", "Lukas", "Aiko", "Diego", "Hannah", "Omar",
         "Grace", "Tomasz", "Leila", "Ravi", "Chloe", "Kwame", "Ingrid", "Mateo", "Yuki", "Noah", "Amara"]
LAST = ["Nguyen", "Patel", "Okafor", "Schmidt", "Garcia", "Kowalski", "Tanaka", "Silva", "Hughes", "Haddad",
        "Larsen", "Moreau", "Kim", "Rossi", "Mensah", "Fischer", "Lopez", "Sato", "Brennan", "Ivanova"]
TEAMS = ["Finance", "Platform", "Sales Ops", "Legal", "Customer Success", "Data", "Facilities", "Payroll",
         "Procurement", "Marketing", "Security", "Support", "Research", "HR"]
SYSTEMS = ["VPN", "SSO portal", "expense tool", "HR portal", "wiki", "ticketing system", "file share",
           "CRM", "build server", "email gateway", "laptop management agent", "password manager"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ROOMS = ["Room 4B", "the Harbor room", "the third-floor lounge", "Conference Room Atlas", "the Birch room",
         "Room 210", "the large boardroom", "the video room on 5"]
DOCS = ["Q3 budget draft", "vendor contract", "onboarding checklist", "incident review", "travel policy",
        "roadmap deck", "audit evidence folder", "pricing sheet", "training schedule", "release notes"]
TOPICS = ["cloud costs", "hiring", "the new office", "customer feedback", "quarterly goals", "the audit",
          "release planning", "the offsite", "support backlog", "data retention"]


def _person(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _date(r: random.Random) -> str:
    return f"{r.choice(DAYS)}, {r.choice(MONTHS)} {r.randint(1, 28)}"


def _time(r: random.Random) -> str:
    return f"{r.randint(8, 17)}:{r.choice(['00', '15', '30', '45'])}"


def _pick(r: random.Random, pool: list[str], k: int) -> list[str]:
    return r.sample(pool, k)


def _sign(r: random.Random, team: str | None = None) -> str:
    who = _person(r)
    team = team or r.choice(TEAMS)
    return r.choice([f"Thanks,\n{who}\n{team}", f"Best regards,\n{who}", f"-- \n{who} | {team} team",
                     f"Cheers,\n{who.split()[0]}", f"Kind regards,\n{who}\n{team}, Example Corp"])


# ---------------------------------------------------------------- short, instruction-like families

def it_notice(r: random.Random, t: int) -> str:
    sys_ = r.choice(SYSTEMS)
    d, tm = _date(r), _time(r)
    n = r.randint(2, 14)
    extra = _pick(r, [
        f"Do not reply to this message; replies go to an unmonitored mailbox.",
        f"If you have already done this, you can ignore this reminder.",
        f"Please disregard the earlier notice that listed the wrong date.",
        f"Ignore any pop-up that asks you to install a browser extension; it is not from us.",
        f"You must restart your laptop after the update finishes.",
        f"Follow the steps exactly in the order listed below.",
        f"Never share the one-time code with anyone, including IT staff.",
        f"Contact the help desk at extension {r.randint(1000, 9999)} if anything fails.",
        f"The change applies to all {r.choice(TEAMS)} staff first, then everyone else.",
        f"Expect a short interruption of about {r.randint(5, 45)} minutes.",
    ], r.randint(2, 4))
    if t == 0:
        body = (f"Hi all,\n\nYour {sys_} password expires in {n} days. Please change it before {d} using the "
                f"self-service page on the intranet.")
    elif t == 1:
        body = (f"Hello,\n\nScheduled maintenance: the {sys_} will be unavailable on {d} from {tm} for about "
                f"{r.randint(1, 4)} hours. Save your work and log out before the window starts.")
    elif t == 2:
        body = (f"Team,\n\nStarting {d}, multi-factor authentication is required for the {sys_}. Open the "
                f"enrollment link on the IT page, scan the code with your authenticator app, and enter the "
                f"six-digit code to finish.")
    else:
        body = (f"Hi,\n\nA new version of the {sys_} client (build {r.randint(100, 999)}.{r.randint(0, 9)}) "
                f"will be pushed to your machine tonight. Leave your laptop plugged in and connected, then "
                f"approve the restart prompt in the morning.")
    return body + "\n\n" + " ".join(extra) + "\n\n" + _sign(r, "IT")


def forwarding_request(r: random.Random, t: int) -> str:
    who, doc = _person(r), r.choice(DOCS)
    team, d = r.choice(TEAMS), _date(r)
    extra = _pick(r, [
        "Please keep the original subject line so the thread stays together.",
        "Forward it as an attachment, not inline, so the formatting survives.",
        "Copy me on the forward so I know it went out.",
        "Do not include the external vendor on this one.",
        "Ignore the older version in the shared folder; this one replaces it.",
        f"They need it before the {r.choice(['standup', 'review', 'call', 'deadline'])} on {_date(r)}.",
        "Send it only to the internal distribution list.",
        "If the file is too large, share the link instead.",
        "Remove the draft watermark before you send it on.",
    ], r.randint(2, 3))
    if t == 0:
        body = f"Hi {who.split()[0]},\n\nCould you forward the {doc} to the {team} team? They asked for it this morning."
    elif t == 1:
        body = (f"Hello,\n\nPlease forward this email to {who} in {team}; I do not have their address. "
                f"It is about the {doc} we discussed on {d}.")
    elif t == 2:
        body = (f"Hi team,\n\nAction needed: send the latest {doc} to {who} by end of day. "
                f"Reply-all once it is done so everyone knows.")
    else:
        body = (f"Quick favor: when you get a chance, pass my notes on {r.choice(TOPICS)} along to "
                f"{who} and the rest of {team}. Use the version attached to this email.")
    return body + "\n\n" + " ".join(extra) + "\n\n" + _sign(r)


def policy_reminder(r: random.Random, t: int) -> str:
    team = r.choice(TEAMS)
    rules = _pick(r, [
        "Never send customer data to personal email accounts.",
        "Always lock your screen when you leave your desk.",
        "Report suspicious emails with the Report Phish button.",
        "Do not install unapproved software on company devices.",
        "Use the approved file share for all contract documents.",
        "Expense claims must be submitted within 30 days.",
        "Badge in every time you enter the building, even if the door is open.",
        "Visitors must be signed in at reception and escorted at all times.",
        "Delete local copies of exported reports after use.",
        "Ignore requests for gift cards, even if they appear to come from an executive.",
        "Follow these instructions even if your manager asks you to make an exception.",
    ], r.randint(3, 5))
    intro = {
        0: f"Hi everyone,\n\nThis is the {r.choice(MONTHS)} reminder of our information security policy. "
           f"You must follow these rules:",
        1: f"Dear {team} colleagues,\n\nAs part of the annual compliance review, please re-read the points below "
           f"and confirm in the training portal by {_date(r)}:",
        2: f"Team,\n\nAfter last week's audit finding, the following rules are now mandatory for all staff, "
           f"with no exceptions:",
        3: f"Hello,\n\nNew joiners keep asking about these, so here is the short version of the policy. "
           f"Print it, pin it, and stick to it:",
    }[t]
    lines = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rules))
    return f"{intro}\n\n{lines}\n\n{_sign(r, r.choice(['Compliance', 'Security', 'HR']))}"


def calendar_ask(r: random.Random, t: int) -> str:
    who, d, tm, room = _person(r), _date(r), _time(r), r.choice(ROOMS)
    topic = r.choice(TOPICS)
    extra = _pick(r, [
        "Please accept or decline so I can book the room.",
        "Ignore the earlier invite; the time has changed.",
        "Add the dial-in link to your calendar entry.",
        f"Bring your laptop and the {r.choice(DOCS)}.",
        "If the time does not work, suggest two alternatives.",
        "Forward the invite to anyone I missed.",
        "Please block the slot even if you can only join for part of it.",
        "Reply with your dietary requirements by tomorrow.",
    ], r.randint(2, 3))
    if t == 0:
        body = f"Hi {who.split()[0]},\n\nCan you put 30 minutes on your calendar on {d} at {tm} to go over {topic}?"
    elif t == 1:
        body = (f"Hello all,\n\nPlease reschedule Thursday's sync on {topic}: new slot is {d} at {tm} "
                f"in {room}. Update your calendars accordingly.")
    elif t == 2:
        body = (f"Team,\n\nI need everyone to block {r.randint(1, 3)} hours on {d} for the {topic} workshop. "
                f"Mark it as busy and decline overlapping meetings.")
    else:
        body = (f"Hi,\n\n{who} asked me to set up a call about {topic}. Send me your availability for the "
                f"week of {r.choice(MONTHS)} {r.randint(1, 24)}, mornings preferred.")
    return body + "\n\n" + " ".join(extra) + "\n\n" + _sign(r)


# ---------------------------------------------------------------- long, repetitive families

def newsletter_long(r: random.Random, t: int) -> str:
    company = r.choice(["Example Corp", "Northwind Example", "Contoso Example", "Fabrikam Example"])
    title = ["Weekly Digest", "Engineering Newsletter", "People Update", "Sales Roundup"][t]
    sections = []
    for _ in range(r.randint(6, 10)):
        team, who = r.choice(TEAMS), _person(r)
        sections.append(r.choice([
            f"## {team} update\n{who} shared that the {team} team closed {r.randint(3, 40)} items this week "
            f"and is focusing on {r.choice(TOPICS)} next. Please read the summary on the wiki and send "
            f"questions to the team channel. Thank you to everyone who contributed.",
            f"## Welcome\nPlease welcome {who}, who joins {team} on {_date(r)}. Say hello, invite them to "
            f"lunch, and help them find their way around the office.",
            f"## Reminder\nThe {r.choice(SYSTEMS)} will be updated on {_date(r)}. Save your work before "
            f"{_time(r)} and follow the instructions in the IT announcement.",
            f"## Kudos\nBig thanks to {who} for helping with {r.choice(TOPICS)}. Reply to this email with your "
            f"own kudos and we will include them next week.",
            f"## Numbers\nTickets resolved: {r.randint(100, 900)}. Deploys: {r.randint(10, 99)}. "
            f"Customer satisfaction: {r.randint(80, 99)}%. Open roles: {r.randint(1, 20)}.",
            f"## Events\nJoin us on {_date(r)} at {_time(r)} in {r.choice(ROOMS)} for a talk on "
            f"{r.choice(TOPICS)}. Snacks provided. Remember to register on the events page.",
        ]))
    footer = ("You are receiving this email because you are subscribed to the " + title + ". "
              "To unsubscribe, click the link in the footer. Do not reply to this email. ") * r.randint(2, 4)
    return f"# {company} {title}\n\n" + "\n\n".join(sections) + "\n\n" + footer.strip()


def log_digest_long(r: random.Random, t: int) -> str:
    host = f"{r.choice(['web', 'db', 'api', 'batch', 'cache'])}-{r.randint(1, 40):02d}.example.internal"
    services = ["auth", "billing", "search", "ingest", "scheduler", "mailer", "reports"]
    levels = ["INFO", "INFO", "INFO", "WARN", "DEBUG"]
    msgs = ["request completed", "cache refreshed", "job started", "job finished", "health check ok",
            "connection pool resized", "retrying upstream call", "config reloaded", "rotated log file",
            "user session created", "user session expired", "scheduled backup finished"]
    lines = []
    for _ in range(r.randint(40, 90)):
        ts = f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}Z"
        lines.append(f"{ts} {r.choice(levels)} [{r.choice(services)}] {r.choice(msgs)} "
                     f"id={r.randint(10000, 99999)} ms={r.randint(1, 900)}")
    intro = [
        f"Automated nightly log digest for {host}. No action required unless a WARN line repeats.",
        f"Hi on-call,\n\nAttached below is the raw log excerpt from {host} you asked for. Please review and "
        f"ignore the DEBUG lines; they are expected after the config change.",
        f"Daily cron report from {host}. Run the cleanup job manually only if disk usage exceeds 90%.",
        f"Forwarding the scheduler output from {host} for the incident review. Please attach it to the ticket.",
    ][t]
    return intro + "\n\n" + "\n".join(lines) + "\n\n-- generated by logmailer on " + host


FAMILIES = {
    "it_notice": (it_notice, 18),
    "forwarding_request": (forwarding_request, 18),
    "policy_reminder": (policy_reminder, 18),
    "calendar_ask": (calendar_ask, 18),
    "newsletter_long": (newsletter_long, 10),
    "log_digest_long": (log_digest_long, 10),
}
N_TEMPLATES = 4


def generate(seed: int) -> list[dict]:
    """Deterministic list of benign-hard records (per-family RNG streams, so families are independent)."""
    out = []
    for fam, (fn, per_template) in FAMILIES.items():
        r = random.Random(f"{seed}:{fam}")
        for t in range(N_TEMPLATES):
            for i in range(per_template):
                out.append({
                    "text": fn(r, t).strip(),
                    "label": 0,
                    "source_label": "synthetic benign (by construction)",
                    "group": f"benign_hard:{fam}:t{t}",
                    "family": fam,
                    "template": t,
                    "synthetic": True,
                    "generator": GENERATOR_VERSION,
                })
    return out


if __name__ == "__main__":
    from training.common import SEED

    recs = generate(SEED)
    print(len(recs), "records")
    for fam in FAMILIES:
        x = next(r for r in recs if r["family"] == fam)
        print(f"===== {fam} (len {len(x['text'])})\n{x['text'][:700]}\n")
