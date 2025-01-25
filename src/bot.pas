{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit bot;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, sdl2, sdl.timer, sdl.app.stimuli.contract;

type

  { TParticipantBot }

  TParticipantBot = class(TComponent)
  private
    FFakeExpectedResponseTimer : TSDLTimer;
    FTargetStimulus : IStimuli;
    procedure Response(Sender : TObject);
  public
    constructor Create(AOwner : TComponent); override;
    destructor Destroy; override;
    procedure Start(AStimulus : IStimuli);
    procedure Stop;
  end;

const
  SESSION_BOT = SDL_USEREVENT+3;

var
  ParticipantBot : TParticipantBot;
  CheatsModeOn : Boolean;

resourcestring
  RSErrorUnknownTarget = 'TParticipantBot.Async: target not assigned';

{ Two Choice Porcentage Bias
r := Random;
if r < (TrackBarRandomBias.Position/100) then
  repeat
    ri := Random(10);
  until not Odd(ri)
else
  repeat
    ri := Random(10);
  until Odd(ri);
choice := choices[ri];
}

implementation

{ TBot }

procedure TParticipantBot.Response(Sender : TObject);
begin
  if FFakeExpectedResponseTimer.Enabled then
    FTargetStimulus.DoExpectedResponse;
end;

constructor TParticipantBot.Create(AOwner : TComponent);
begin
  inherited Create(AOwner);
  FFakeExpectedResponseTimer := TSDLTimer.Create;
  FFakeExpectedResponseTimer.Interval := 1000;
  FFakeExpectedResponseTimer.OnTimer := @Response;
end;

destructor TParticipantBot.Destroy;
begin
  FFakeExpectedResponseTimer.Free;
  inherited Destroy;
end;

procedure TParticipantBot.Start(AStimulus : IStimuli);
begin
  FTargetStimulus := AStimulus;
  FFakeExpectedResponseTimer.Start;
end;

procedure TParticipantBot.Stop;
begin
  FFakeExpectedResponseTimer.Stop;
end;

end.

