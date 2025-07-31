{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, session.blocks, sdl.timer;

type

  { TSession }

  TSession = class(TComponent)
  private
    FTimer : TSDLTimer;
    FBlock : TBlock;
    FOnBeforeStart: TNotifyEvent;
    FOnEndSession: TNotifyEvent;
    function GetOnHitCriteriaAtSessionEnd: TNotifyEvent;
    function GetOnNotHitCriteriaAtSessionEnd: TNotifyEvent;
    procedure PlayBlock;
    procedure EndBlock(Sender : TObject);
    procedure SetOnBeforeStart(AValue: TNotifyEvent);
    procedure SetOnEndSession(AValue: TNotifyEvent);
    procedure SetOnHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
    procedure SetOnNotHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
    procedure TimerOnTimer(Sender: TObject);
  public
    constructor Create(AOwner : TComponent); override;
    destructor Destroy; override;
    procedure EndSession;
    procedure Play;
    property OnEndSession : TNotifyEvent read FOnEndSession write SetOnEndSession;
    property OnHitCriteriaAtSessionEnd  : TNotifyEvent read GetOnHitCriteriaAtSessionEnd write SetOnHitCriteriaAtSessionEnd;
    property OnNotHitCriteriaAtSessionEnd : TNotifyEvent read GetOnNotHitCriteriaAtSessionEnd write SetOnNotHitCriteriaAtSessionEnd;
    property OnBeforeStart : TNotifyEvent read FOnBeforeStart write SetOnBeforeStart;
    property Timer : TSDLTimer read FTimer;
  end;

var
  SDLSession: TSession;

implementation

uses
  timestamps
  , session.counters
  , session.pool
  , session.configurationfile
  , session.endcriteria
  , sdl.app.trials.factory
  ;

{ TSession }

procedure TSession.PlayBlock;
begin
  if EndCriteria.OfSession then begin
    EndSession;
  end else begin
    FBlock.BeforePlay;
    FBlock.Play;
  end;
end;

function TSession.GetOnHitCriteriaAtSessionEnd: TNotifyEvent;
begin
  Result := EndCriteria.OnHitCriteriaAtSessionEnd;
end;

function TSession.GetOnNotHitCriteriaAtSessionEnd: TNotifyEvent;
begin
  Result := EndCriteria.OnNotHitCriteriaAtSessionEnd;
end;

procedure TSession.EndBlock(Sender: TObject);
begin
  PlayBlock;
end;

procedure TSession.EndSession;
begin
  TTrialFactory.GetLastTrial.Show;
  if Assigned(OnEndSession) then OnEndSession(Self);
end;

procedure TSession.SetOnBeforeStart(AValue: TNotifyEvent);
begin
  if FOnBeforeStart=AValue then Exit;
  FOnBeforeStart:=AValue;
end;

procedure TSession.SetOnEndSession(AValue: TNotifyEvent);
begin
  if FOnEndSession=AValue then Exit;
  FOnEndSession:=AValue;
end;

procedure TSession.SetOnHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
begin
  EndCriteria.OnHitCriteriaAtSessionEnd := AValue;
end;

procedure TSession.SetOnNotHitCriteriaAtSessionEnd(AValue: TNotifyEvent);
begin
  EndCriteria.OnNotHitCriteriaAtSessionEnd := AValue;
end;

procedure TSession.TimerOnTimer(Sender: TObject);
begin
  FTimer.Stop;
  EndSession;
end;

constructor TSession.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  Pool.Counters.BeforeBeginSession;
  EndCriteria := TEndCriteria.Create;
  Pool.EndCriteria := EndCriteria;
  FTimer := TSDLTimer.Create;
  FTimer.OnTimer:=@TimerOnTimer;
  FTimer.Interval := 0;
  FBlock := TBlock.Create(Self);
  FBlock.OnEndBlock := @EndBlock;
end;

destructor TSession.Destroy;
begin
  Pool.Counters.BeforeEndSession;
  EndCriteria.Free;
  FTimer.Free;
  inherited Destroy;
end;

procedure TSession.Play;
begin
  if Assigned(OnBeforeStart) then OnBeforeStart(Self);
  if FTimer.Interval > 0 then
    FTimer.Start;
  PlayBlock;
end;

end.

