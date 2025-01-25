{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.renderer.thread;

{$mode objfpc}{$H+}

interface

uses
  Classes;

type

  { TRendererThread }

  TRendererThread = class (TThread)
  strict private
    FMsg: string;
    FShouldStop : Boolean;
    FRTLEventMainThread : PRTLEvent;
    procedure PrintMessage;
    procedure DoInvalidate;
  protected
    procedure Execute; override;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Stop;
    procedure Close;
    procedure Render;
    procedure Log(const AMsg: string; AppendLineEnd: boolean = true);
    property Msg : string read FMsg write FMsg;
  end;

implementation

uses
  SysUtils,
  //sdl.app.output,
  sdl.app.renderer.types,
  sdl.app.renderer.validation,
  session.parameters.global;

const
  GFPS = 1000 div 30;

var
  GCriticalSection : TRTLCriticalSection;

procedure TRendererThread.Execute;
  procedure DoWork;
  var
    LShouldStop : Boolean;
  begin
    while not Terminated do begin
      EnterCriticalSection(GCriticalSection);
      LShouldStop := FShouldStop;
      LeaveCriticalSection(GCriticalSection);

      if LShouldStop then begin
        Break;
      end;

      Sleep(DELTA_TIME);
      Queue(@DoInvalidate);
    end;
  end;
begin
  NameThreadForDebugging(ClassName);
  FRTLEventMainThread := RTLEventCreate;
  while not Terminated do begin
    RTLEventWaitFor(FRTLEventMainThread);
    DoWork;
  end;
  Log(ClassName+': Terminated ...');
end;

constructor TRendererThread.Create;
begin
  inherited Create(False);
  FreeOnTerminate := True;
end;

destructor TRendererThread.Destroy;
begin

  inherited Destroy;
end;

procedure TRendererThread.Stop;
begin
  FShouldStop := True;
end;

procedure TRendererThread.Close;
begin
  RTLEventSetEvent(FRTLEventMainThread);
  RTLEventDestroy(FRTLEventMainThread);
end;

procedure TRendererThread.Render;
begin
  FShouldStop := False;
  RTLEventSetEvent(FRTLEventMainThread);
end;

procedure TRendererThread.PrintMessage;
begin
  //Print(Msg);
end;

procedure TRendererThread.DoInvalidate;
begin
  GPaintingInvalidated := True;
end;

procedure TRendererThread.Log(const AMsg: string; AppendLineEnd: boolean);
var
  s: String;
begin
  EnterCriticalsection(GCriticalSection);
  s:=AMsg;
  if AppendLineEnd then
    s:=s+LineEnding;
  Msg:=s;
  Synchronize(@PrintMessage);
  LeaveCriticalsection(GCriticalSection);
end;

initialization
  InitCriticalSection(GCriticalSection);

finalization
  DoneCriticalsection(GCriticalSection);


end.
