{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.mts;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils
  , sdl.app.trials
  , sdl.app.stimuli.contract
  , sdl.app.stimuli.mts
  //, sdl.app.navigable.contract
  ;

type

  { TMTS }

  TMTS = class sealed (TTrial)
    private
      FStimuli : TMTSStimuli;
      //procedure DoInvalidate(Sender: TObject);
    protected
      function GetIStimuli: IStimuli; override;
    public
      constructor Create; override;
      destructor Destroy; override;
      procedure EndTrial; override;
      procedure Show; override;
      procedure Hide; override;
  end;


implementation

{ TMTS }

constructor TMTS.Create;
begin
  inherited Create;
  FStimuli := TMTSStimuli.Create;
  FStimuli.Trial:= Self;
  //FStimuli.OnResponse := @DoInvalidate;
  FStimuli.OnFinalize := @EndTrialCallBack;
end;

destructor TMTS.Destroy;
begin
  { free stuff }
  FStimuli.Free;
  inherited Destroy;
end;

//procedure TMTS.DoInvalidate(Sender: TObject);
//begin
//  Invalidate;
//end;

function TMTS.GetIStimuli: IStimuli;
begin
  Result := FStimuli.AsIStimuli;
end;

procedure TMTS.EndTrial;
begin
  inherited EndTrial;
end;

procedure TMTS.Show;
begin
  inherited Show;
  //FIStimuli.Start;
end;

procedure TMTS.Hide;
begin
  inherited Hide;
  //FIStimuli.Stop;
end;

end.

